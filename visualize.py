import os
import numpy as np
import json
from PIL import Image


# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './data/hw02_preds'
anno_path = './data/hw02_annotations'

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# save preds (overwrites any previous predictions!)


with open(os.path.join(anno_path,'annotations_train.json'),'r') as f_a:
    data_a = json.load(f_a)
    with open(os.path.join(preds_path,'preds_train_redlight1.json'),'r') as f_t:
        data_t = json.load(f_t)

        # Visualiza all the results
        count = 0
        for img_name in data_a:
            I = Image.open(os.path.join(data_path,img_name))
            I = np.array(I)
            h, w, _ = I.shape
            bounding_boxes = data_a[img_name]
            for b in bounding_boxes:
                [tl_row,tl_col,br_row,br_col] = b[:4]
                tl_row = int(tl_row)
                tl_col = int(tl_col)
                br_row = int(min(br_row,h))
                br_col = int(min(br_col,w))
                I[tl_row:br_row, tl_col, :] = [0,255,0]
                I[tl_row:br_row, br_col-1, :] = [0,255,0]
                I[tl_row, tl_col:br_col, :] = [0,255,0]
                I[br_row-1, tl_col:br_col, :] = [0,255,0]

            bounding_boxes = data_t[img_name]
            for b in bounding_boxes:
                [tl_row,tl_col,br_row,br_col] = b[:4]
                tl_row = int(tl_row)
                tl_col = int(tl_col)
                br_row = int(min(br_row,h))
                br_col = int(min(br_col,w))
                I[tl_row:br_row, tl_col, :] = [0,0,255]
                I[tl_row:br_row, br_col-1, :] = [0,0,255]
                I[tl_row, tl_col:br_col, :] = [0,0,255]
                I[br_row-1, tl_col:br_col, :] = [0,0,255]
            img = Image.fromarray(I, 'RGB')
            img.save(os.path.join(preds_path,img_name))

            count += 1
            if count > 10:
                break
