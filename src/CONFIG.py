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
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LEN = 512  # Giữ nguyên như yêu cầu
    BATCH_SIZE = 2  # Batch size nhỏ để fit memory
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 * 4 = 16
    NUM_EPOCHS = 100 
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    WARMUP_STEPS = 0
    EARLY_STOPPING_PATIENCE = 4
    EARLY_STOPPING_DELTA = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42
    AUGMENT = False


class LSTMConfig:
    VOCAB_SIZE = 50000  # Will be set dynamically based on dataset
    EMBEDDING_DIM = 256  # Reduced for less overfitting
    HIDDEN_DIM = 512     # Reduced for better generalization
    OUTPUT_DIM = 1
    NUM_LAYERS = 2       # Reduced layers to prevent overfitting
    
    # Training parameters - more conservative
    BATCH_SIZE = 16      # Smaller batch for better generalization
    NUM_EPOCHS = 50      # Reduced epochs
    LEARNING_RATE = 5e-4 # Lower learning rate for stability
    WEIGHT_DECAY = 1e-3  # Increased weight decay for regularization
    
    # Stronger regularization
    EMBEDDING_DROPOUT = 0.3
    LSTM_DROPOUT = 0.4
    CLASSIFIER_DROPOUT = 0.5
    
    # Architecture parameters
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    NUM_RESIDUAL_BLOCKS = 3  # Reduced blocks
    RESIDUAL_DROPOUT = 0.4   # Increased dropout
    
    # Early stopping - more aggressive
    EARLY_STOPPING_PATIENCE = 7  # Reduced patience
    EARLY_STOPPING_DELTA = 0.005 # Increased delta for stricter improvement
    
    # Device and optimization
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42
    
    # Multi-GPU
    USE_MULTI_GPU = True
    
    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3