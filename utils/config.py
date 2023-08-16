import os

# Directory settings
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = '/mnt/d/ubuntu/datasets/sr_benchmark/'
LOG_DIR = 'logs/'
TEST_RESULTS_DIR = 'test_results/'

DATASET_PATH = {'train' : DATASET_DIR + 'Data/train/',
                'test' : DATASET_DIR + 'Data/valid_ipt/',
                'evaluation' : DATASET_DIR + 'Data/valid_ipt/',
                'valid' : DATASET_DIR + 'Data/valid_ipt/'}

DATA_LIST_DIR = DATASET_DIR + 'Data/DataName/'

TRAINING_DATA_LIST = ["DIV2K", "Flickr2K"]

# Image settings
PIXEL_MAX_VALUE = 255

IMG_NORM_MEAN = [0.406, 0.456, 0.485]  # BGR order
IMG_NORM_STD = [0.225, 0.224, 0.229]  # BGR order

GRAY_COEF = [24.966, 128.553, 65.481]  # BGR order
GRAY_BIAS = 16.

COLOR_SHIFTING_RANGE = 10

# Validation settings
PIXEL_INTERSECTION = 17