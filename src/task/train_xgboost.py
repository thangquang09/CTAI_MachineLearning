import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from data_utils.load_data import Get_Loader
from model.init_model import build_model
from eval_metric.evaluate import ScoreCalculator
from data_utils.load_data import create_ans_space
from tqdm import tqdm

class XGBoost_NLI_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric = config['train']['metric_for_best_model']
        self.weight_decay = config['train']['weight_decay']
        self.answer_space = create_ans_space(config)
        self.dataloader = Get_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = build_model(config, self.answer_space)
        
        if torch.cuda.device_count() > 1:
            self.base_model = nn.DataParallel(self.base_model, device_ids=[0, 1])
        self.base_model = self.base_model.to(self.device)
        self.compute_score = ScoreCalculator()
        
        # Only optimize the neural network parts (text embedding + attention)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
    
    def extract_features_and_labels(self, dataloader):
        """Extract features using neural network components"""
        self.base_model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for sent1, sent2, labels, id in tqdm(dataloader, desc="Extracting features"):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    features = self.base_model._extract_features(sent1, sent2)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.vstack(all_features), np.concatenate(all_labels)
    
    def training(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
        train_loader, valid_loader = self.dataloader.load_train_dev()

        # Check for existing checkpoints
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold = 0
        
        # Phase 1: Train neural network components (feature extraction)
        print("Phase 1: Training neural network components...")
        self.base_model.train()
        
        for epoch in range(initial_epoch, min(10, self.num_epochs + initial_epoch)):  # First 10 epochs for NN
            train_loss = 0.
            
            for it, (sent1, sent2, labels, id) in enumerate(tqdm(train_loader, desc=f"NN Epoch {epoch+1}")):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits, loss = self.base_model(sent1, sent2, labels.to(self.device))
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                train_loss += loss
            
            self.scheduler.step()
            train_loss /= len(train_loader)
            print(f"NN training epoch {epoch + 1}, loss: {train_loss:.4f}")
        
        # Phase 2: Extract features and train XGBoost
        print("Phase 2: Extracting features and training XGBoost...")
        
        # Extract features from training data
        train_features, train_labels = self.extract_features_and_labels(train_loader)
        valid_features, valid_labels = self.extract_features_and_labels(valid_loader)
        
        # Train XGBoost classifier
        if hasattr(self.base_model, 'module'):  # DataParallel case
            self.base_model.module.fit_xgboost(train_features, train_labels)
        else:
            self.base_model.fit_xgboost(train_features, train_labels)
        
        # Evaluate XGBoost
        self.base_model.eval()
        valid_acc = 0.
        valid_f1 = 0.
        
        with torch.no_grad():
            for it, (sent1, sent2, labels, id) in enumerate(tqdm(valid_loader, desc="Evaluating XGBoost")):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.base_model(sent1, sent2)
                preds = torch.argmax(logits, dim=-1)
                valid_acc += self.compute_score.acc(labels, preds)
                valid_f1 += self.compute_score.f1(labels, preds)
                
        valid_acc /= len(valid_loader)
        valid_f1 /= len(valid_loader)
        
        print(f"XGBoost validation acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}")
        
        # Save final model
        if self.best_metric == 'accuracy':
            score = valid_acc
        elif self.best_metric == 'f1':
            score = valid_f1
        
        torch.save({
            'epoch': 0,  # XGBoost doesn't have epochs
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score
        }, os.path.join(self.save_path, 'best_model.pth'))
        
        print(f"Saved XGBoost model with {self.best_metric} of {score:.4f}")
        
        with open('log.txt', 'a') as file:
            file.write(f"XGBoost training completed\n")
            file.write(f"Final validation acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}\n")
