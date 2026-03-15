# peak_finder/config.py
"""Simple configuration"""

class Config:
    # Data
    XRD_LENGTH = 3500
    WINDOW_SIZE = 51  # Window size for peak detection
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Data sampling
    POSITIVE_RATIO = 0.6  # Ratio of positive samples in each batch
    SAMPLES_PER_XRD = 100  # Number of windows sampled per XRD per epoch
    
    # Loss function
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    USE_POSITION_WEIGHT = True  # Use position weight (peaks at the beginning are more important)
    
    # Inference
    STRIDE = 1  # Sliding window stride
    THRESHOLD = 0.8  # Threshold for peak detection
    
    # Paths 
    # DATA_PATH = '/scratch/ksu4/PXRD-GPT/xrd_simplified_3500_55612_noisepxrd.csv'
    # training_data_file = '/scratch/ksu4/training_set_CNN_3500_predict_peak_V4_spgjson.csv'
    # testing_data_file = '/scratch/ksu4/testing_set_CNN_3500_predict_peak_V4_spgjson.csv'
    SAVE_DIR = '/checkpoints_sampele100_ratio0.6_windowsize51_noisepeak/'
    #SAVE_DIR = '/scratch/ksu4/checkpoints_30_0.4_51/'
    #SAVE_DIR = '/scratch/ksu4/checkpoints_sampele100_ratio0.5_windowsize41/'
    #SAVE_DIR = '/scratch/ksu4/checkpoints_sampele100_ratio0.5_windowsize21/'
