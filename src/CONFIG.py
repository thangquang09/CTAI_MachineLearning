import torch

CASE = 1
BATCH_SIZE = 128

EMBEDDING_DIM = 512
HIDDEN_DIM = 512
OUTPUT_DIM = 1 # Output 1 giá trị logit cho binary classification
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
SEQ_LENGTH = 600
EARLY_STOPPING = False  # Tắt early stopping để train full epochs


class PretrainedModelConfig:
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_LEN = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    WARMUP_STEPS = 0
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_DELTA = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42