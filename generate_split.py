import numpy as np
import os

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = './data/RedLights2011_Medium'
gts_path = './data/hw02_annotations'
split_path = './data/hw02_splits'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

split_test = False # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below.
'''

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'annotations.json'),'r') as f:
        gts = json.load(f)

    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below.
    '''

    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
