from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import pickle
import os
from text_module.init_text_embedding import build_text_embedding

class XGBoost_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
        super(XGBoost_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout = config["model"]["dropout"]
        self.max_length = config['tokenizer']['max_length']
        
        # XGBoost specific parameters
        self.n_estimators = config['xgboost']['n_estimators']
        self.max_depth = config['xgboost']['max_depth']
        self.learning_rate = config['xgboost']['learning_rate']
        self.subsample = config['xgboost']['subsample']
        self.colsample_bytree = config['xgboost']['colsample_bytree']
        self.reg_alpha = config['xgboost']['reg_alpha']
        self.reg_lambda = config['xgboost']['reg_lambda']
        
        self.text_embedding = build_text_embedding(config)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # XGBoost classifier will be initialized during first training
        self.xgb_classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # For loss calculation during training
        self.criterion = nn.CrossEntropyLoss()

    def _extract_features(self, id1_text: List[str], id2_text: List[str]):
        """Extract features using text embedding and attention"""
        embedded, mask = self.text_embedding(id1_text, id2_text)
        feature_attended = self.attention_weights(torch.tanh(embedded))
        
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * embedded, dim=1)
        
        # Apply dropout
        feature_attended = self.dropout_layer(feature_attended)
        
        return feature_attended

    def fit_xgboost(self, train_features, train_labels):
        """Fit XGBoost classifier with extracted features (without early stopping)"""
        # Convert to numpy if tensor
        if isinstance(train_features, torch.Tensor):
            train_features = train_features.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.detach().cpu().numpy()
            
        # Initialize XGBoost classifier
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss' if self.num_labels > 2 else 'logloss'
        )
        
        # Fit the classifier
        self.xgb_classifier.fit(train_features, train_labels)

    def fit_xgboost_with_early_stopping(self, train_features, train_labels, valid_features, valid_labels, patience=5):
        """Fit XGBoost classifier with early stopping"""
        # Convert to numpy if tensor
        if isinstance(train_features, torch.Tensor):
            train_features = train_features.detach().cpu().numpy()
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.detach().cpu().numpy()
        if isinstance(valid_features, torch.Tensor):
            valid_features = valid_features.detach().cpu().numpy()
        if isinstance(valid_labels, torch.Tensor):
            valid_labels = valid_labels.detach().cpu().numpy()
            
        # Initialize XGBoost classifier with more estimators for early stopping
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators * 3,  # Increase for early stopping
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss' if self.num_labels > 2 else 'logloss',
            early_stopping_rounds=patience
        )
        
        # Fit with early stopping
        print(f"Training XGBoost with early stopping (patience={patience})...")
        self.xgb_classifier.fit(
            train_features, train_labels,
            eval_set=[(valid_features, valid_labels)],
            verbose=True
        )
        
        print(f"XGBoost training stopped at {self.xgb_classifier.best_iteration + 1} iterations")
        print(f"Best validation score: {self.xgb_classifier.best_score:.4f}")

    def save_xgboost_classifier(self, save_path: str):
        """Save the trained XGBoost classifier"""
        if self.xgb_classifier is not None:
            xgb_path = os.path.join(save_path, 'xgboost_classifier.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_classifier, f)
            print(f"XGBoost classifier saved to {xgb_path}")

    def load_xgboost_classifier(self, save_path: str):
        """Load the trained XGBoost classifier"""
        xgb_path = os.path.join(save_path, 'xgboost_classifier.pkl')
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                self.xgb_classifier = pickle.load(f)
            print(f"XGBoost classifier loaded from {xgb_path}")
            return True
        else:
            print(f"XGBoost classifier not found at {xgb_path}")
            return False

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):
        # Extract features
        features = self._extract_features(id1_text, id2_text)
        
        # During training phase
        if self.training and labels is not None:
            # For training, we need to return logits and loss for backprop
            # Create a dummy linear layer for gradient computation
            if not hasattr(self, 'dummy_classifier'):
                self.dummy_classifier = nn.Linear(self.intermediate_dims, self.num_labels).to(self.device)
            
            logits = self.dummy_classifier(features)
            loss = self.criterion(logits, labels)
            return logits, loss
            
        # During inference phase
        else:
            if self.xgb_classifier is None:
                raise ValueError("XGBoost classifier not trained yet! Call fit_xgboost first.")
            
            # Convert features to numpy for XGBoost prediction
            features_np = features.detach().cpu().numpy()
            
            # Get predictions from XGBoost
            if hasattr(self.xgb_classifier, 'predict_proba'):
                probs = self.xgb_classifier.predict_proba(features_np)
                logits = torch.from_numpy(probs).to(self.device)
            else:
                preds = self.xgb_classifier.predict(features_np)
                # Convert to one-hot then to logits
                logits = torch.zeros(len(preds), self.num_labels).to(self.device)
                logits[range(len(preds)), preds] = 1.0
            
            if labels is not None:
                loss = self.criterion(logits, labels)
                return logits, loss
            else:
                return logits

def createXGBoost_Model(config: Dict, answer_space: List[str]) -> XGBoost_Model:
    return XGBoost_Model(config, num_labels=len(answer_space))
