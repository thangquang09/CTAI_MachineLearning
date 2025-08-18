import os
import logging
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
import transformers

from model.init_model import get_model
from data_utils.load_data import Get_Loader, create_ans_space


class Predict:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.answer_space = create_ans_space(config)
        self.checkpoint_path = os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.save_path = config["train"]["output_dir"]
        self.model = get_model(config, num_labels=len(self.answer_space))
        self.model.to(self.device)
        self.dataloader = Get_Loader(config)

    def predict_submission(self):
        # tắt logging của transformers
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)

        # Load checkpoint (toàn bộ object, weights_only=False)
        logging.info("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        
        # Load model state dict
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        logging.info("Model loaded successfully")
            
        self.model.to(self.device)
        self.model.eval()
        
        # For XGBoost model, also load the XGBoost classifier
        if self.config['model']['type_model'] == 'xgboost':
            success = self.model.load_xgboost_classifier(self.save_path)
            if not success:
                raise RuntimeError("Failed to load XGBoost classifier. Make sure training completed successfully.")

        # Load test data
        test_loader = self.dataloader.load_test()

        ids: List[int] = []
        submits: List[str] = []

        logging.info("Obtaining predictions...")
        with torch.no_grad():
            for sent1, sent2, id_batch in tqdm(test_loader):
                # convert sang list nếu DataLoader trả tensor
                sent1 = list(sent1) if isinstance(sent1, torch.Tensor) else sent1
                sent2 = list(sent2) if isinstance(sent2, torch.Tensor) else sent2

                # forward
                logits = self.model(sent1, sent2)

                # argmax để lấy nhãn dự đoán
                preds = logits.argmax(dim=-1).cpu().numpy()
                answers = [self.answer_space[i] for i in preds]
                submits.extend(answers)

                # xử lý id
                if isinstance(id_batch, torch.Tensor):
                    ids.extend(id_batch.tolist())
                else:
                    ids.extend(id_batch)

        # lưu file submission
        df = pd.DataFrame({'id': ids, 'label': submits})
        df.to_csv('./submission.csv', index=False)
        logging.info("Submission saved to ./submission.csv")
