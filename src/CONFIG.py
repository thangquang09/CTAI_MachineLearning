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
    EMBEDDING_DIM = 512  # Keep sufficient capacity
    HIDDEN_DIM = 512     # Keep sufficient capacity
    OUTPUT_DIM = 1
    NUM_LAYERS = 2       # Keep simplified
    
    # Training parameters - more aggressive for learning
    BATCH_SIZE = 16      # Good size
    NUM_EPOCHS = 50      # Sufficient epochs
    LEARNING_RATE = 2e-3 # Higher learning rate to overcome underfit
    WEIGHT_DECAY = 1e-5  # Very low weight decay to allow learning
    
    # Reduced regularization for text classification
    EMBEDDING_DROPOUT = 0.1  # Much lower
    LSTM_DROPOUT = 0.2       # Much lower  
    CLASSIFIER_DROPOUT = 0.3 # Much lower
    
    # Architecture parameters
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    NUM_RESIDUAL_BLOCKS = 1  # Very minimal residual blocks
    RESIDUAL_DROPOUT = 0.2   # Lower dropout
    
    # Early stopping - less aggressive
    EARLY_STOPPING_PATIENCE = 10  # More patience
    EARLY_STOPPING_DELTA = 0.001  # Lower delta
    
    # Device and optimization
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42
    
    # Multi-GPU
    USE_MULTI_GPU = True
    
    # Scheduler - more gentle
    SCHEDULER_FACTOR = 0.7   # Less aggressive reduction
    SCHEDULER_PATIENCE = 5   # More patience