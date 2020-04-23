import os
import numpy as np
import json
from PIL import Image


def compute_convolution(I, T, padding=False):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    Both I and T are 3-dim.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    MY CODE
    '''
    t_height, t_width, t_channels = T.shape
    assert(n_channels == t_channels)

    heatmap = np.zeros((n_rows, n_cols))
    for channel in range(n_channels):
    #for channel in range(1):  ## weakened version
        T_sub = T[:,:,channel]
        I_sub = I[:,:,channel]
        T_sub = T_sub / np.linalg.norm(T_sub)

        # padding, but not used in my best algorithm
        if padding:
            top_pad = int((t_height - 1) / 2)
            bot_pad = t_height - 1 - top_pad
            left_pad = int((t_width - 1) / 2)
            right_pad = t_width - 1 - left_pad
            I_sub = np.pad(I_sub, ((top_pad, bot_pad), (left_pad, right_pad)), constant_values=0)

        I_strided = np.lib.stride_tricks.as_strided(I_sub,
            (n_rows, n_cols, T_sub.shape[0], T_sub.shape[1]),
            (I_sub.strides[0], I_sub.strides[1], I_sub.strides[0], I_sub.strides[1]))

        I_norm = np.linalg.norm(I_strided, axis=(-2,-1))
        I_norm = np.repeat(I_norm, T_sub.shape[0]*T_sub.shape[1])
        I_norm = I_norm.reshape(n_rows, n_cols, T_sub.shape[0], T_sub.shape[1])
        I_normed = I_strided / I_norm
        corr = I_normed * T_sub
        corr = np.sum(corr, (-2, -1))

        heatmap += corr / 3
        #heatmap += corr   ## Weakened version

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN MY CODE
    '''

    box_height = 10
    box_width = 10

    # idx = np.argwhere(heatmap > threshold)
    # if idx.shape[0] > 10:
    #     idx = np.argwhere(heatmap > threshold_high)
    # elif idx.shape[0] < 1:
    #     idx = np.argwhere(heatmap > threshold_low)
    idx = np.argwhere(heatmap > threshold_high)
    num_boxes = idx.shape[0]

    used_idx = []
    for i in range(num_boxes):
        tl_row = int(idx[i,0])
        tl_col = int(idx[i,1])
        br_row = tl_row + box_height
        br_col = tl_col + box_width
        score = float(np.mean(heatmap[tl_row:br_row, tl_col:br_col]))

        is_new = 1
        for j in used_idx:
            tl_row0 = int(idx[j,0])
            tl_col0 = int(idx[j,1])
            if abs(tl_row0 - tl_row) <= box_height and abs(tl_col0 - tl_col) <= box_width:
                is_new = 0
                break
        if is_new:
            used_idx.append(i)
            #print(idx[i,:], score)
            output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END MY CODE
    '''
    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # You may use multiple stages and combine the results
    T1 = Image.open('./redlight1.jpg')
    T1 = np.asarray(T1)
    heatmap1 = compute_convolution(I, T1)

    ## I have tried two stages, but the results are not as good.
    # T2 = Image.open('./redlight2.jpg')
    # T2 = np.asarray(T2)
    # heatmap2 = compute_convolution(I, T2)
    #
    # heatmap = (heatmap1 + heatmap2) / 2
    heatmap = heatmap1
    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# load splits:
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True
threshold_high = 0.9
##threshold_high = 0.88

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
