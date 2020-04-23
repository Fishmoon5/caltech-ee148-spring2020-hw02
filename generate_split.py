import numpy as np
import os
import json

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = './data/RedLights2011_Medium'
gts_path = './data/hw02_annotations'
split_path = './data/hw02_splits'
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available
annotation_name = "formatted_annotations_students.json"

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Codes for splitting file names
'''
idx = np.random.choice( len(file_names), int(0.85*len(file_names)), replace=False )
file_names_train = [file_names[i] for i in idx]
file_names_test = [file_names[i] for i in range(len(file_names)) if i not in idx]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, annotation_name),'r') as f:
        gts = json.load(f)

    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below.
    '''
    for fname in file_names_train:
        gts_train[fname] = gts[fname]
    for fname in file_names_test:
        gts_test[fname] = gts[fname]

    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
