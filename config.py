EPOCHS = 50 # changeable
SAVE_CHECKPOINT = 15 # after how many epochs to save a checkpoint
#LOG_EVERY = 5 # log training and validation metrics every `LOG_EVERY` epochs
BATCH_SIZE = 16
DEVICE = 'cuda' # gpu
LR = 0.001
ROOT_PATH = 'datasets/CamVid'

# the classes that we want to train
CLASSES_TO_TRAIN = [
        'animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
        'parkingblock', 'pedestrian', 'road', 'roadshoulder', 'sidewalk',
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall'
        ]

# DEBUG for visualizations
DEBUG = False