import os

BASE_DIR = os.getcwd()  #返回当前进程的工作目录
SAMPLE_DIR = os.path.join(BASE_DIR,'samples')
# initialize the paths to our training and testing CSV files
TRAIN_CSV = "train.csv"
TEST_CSV = "validation.csv"

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 100
BS = 64

# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# encoding = {'APC': 1,
#             'LBB': 2,
#             'NOR': 3,
#             'PAB': 4,
#             'PVC': 5,
#             'RBB': 6,
#             'VEB': 7,
#             'VFB': 8}
encoding = {'N': 1,
             'A': 2,
             '(AFL': 3,
             '(AFIB': 4,
             '(SVTA': 5,
             '(PREX': 6,
             'V': 7,
             '(B': 8,
             '(T': 9,
             '(VT': 10,
             '(IVR': 11,
             '(VFL': 12,
             'F': 13,
             'L': 14,
             'R': 15,
             '(BII': 16,
             '/': 17}
