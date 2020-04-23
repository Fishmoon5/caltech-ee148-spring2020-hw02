import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    (tl_row1, tl_col1, br_row1, br_col1) = box_1
    (tl_row2, tl_col2, br_row2, br_col2) = box_2
    intersection = max(0, min(br_row1,br_row2) - max(tl_row1,tl_row2)) * max(0, min(br_col1,br_col2) - max(tl_col1,tl_col2))
    union = (br_row1-tl_row1)*(br_col1-tl_col1) + (br_row2-tl_row2)*(br_col2-tl_col2) - intersection
    iou = intersection / union
    #print(box_1, box_2, ":", intersection, union, iou)
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file in preds:
        pred = preds[pred_file]
        pred_idx = []
        for j in range(len(pred)):
            if pred[j][4] >= conf_thr:
                pred_idx.append(j)
        M = len(pred_idx)
        gt = gts[pred_file]
        n = len(gt)

        true_positive = 0
        for i in range(len(gt)):
            for j in pred_idx:
                iou = compute_iou(pred[j][:4], gt[i])
                if iou > iou_thr:
                    true_positive += 1
                    break
        TP += true_positive
        FP += M - true_positive
        FN += n - true_positive
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './data/hw02_preds'
gts_path = './data/hw02_annotations'

# load splits:
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True


'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train_redlight1.json'),'r') as f:
#with open(os.path.join(preds_path,'preds_train_redlight1_weakened.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''
    with open(os.path.join(preds_path,'preds_test_redlight1.json'),'r') as f:
    #with open(os.path.join(preds_path,'preds_test_redlight1_weakened.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.
fig, ax = plt.subplots()
for iou_threshold in [0.5, 0.25, 0.75]:
    scores = []
    for fname in preds_train:
        if len(preds_train[fname]) != 0:
            scores += list(np.array(preds_train[fname])[:,4])
    confidence_thrs = np.sort(np.array(scores,dtype=float)) # using (ascending) list of confidence scores as thresholds
    confidence_thrs = confidence_thrs[range(0, len(confidence_thrs), int(len(confidence_thrs)/20))]
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_threshold, conf_thr=conf_thr)

    precisions = []
    recalls = []
    for i in range(len(tp_train)):
        precision = tp_train[i] / (tp_train[i] + fp_train[i])
        recall = tp_train[i] / (tp_train[i] + fn_train[i])
        #print(tp_train[i], fp_train[i], fn_train[i], precision, recall)
        precisions.append(precision)
        recalls.append(recall)

    ax.plot(recalls, precisions, marker='.')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(["IoU_thr=0.5","IoU_thr=0.25","IoU_thr=0.75"])
plt.title("PR curve for the training set of my best algorithm")
#plt.title("PR curve for the training set of the weakened version of my best algorithm")
plt.show()

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
    fig, ax = plt.subplots()
    for iou_threshold in [0.5, 0.25, 0.75]:
        scores = []
        for fname in preds_test:
            if len(preds_test[fname]) != 0:
                scores += list(np.array(preds_test[fname])[:,4])
        confidence_thrs = np.sort(np.array(scores,dtype=float)) # using (ascending) list of confidence scores as thresholds
        #confidence_thrs = confidence_thrs[[0,10,20,30, -3,-2,-1]]
        #print(confidence_thrs)
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_threshold, conf_thr=conf_thr)

        precisions = []
        recalls = []
        for i in range(len(tp_test)):
            precision = tp_test[i] / (tp_test[i] + fp_test[i])
            recall = tp_test[i] / (tp_test[i] + fn_test[i])
            #print(tp_test[i], fp_test[i], fn_test[i], precision, recall)
            precisions.append(precision)
            recalls.append(recall)

        ax.plot(recalls, precisions, marker='o')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(["IoU_thr=0.5","IoU_thr=0.25","IoU_thr=0.75"])
    plt.title("PR curve for the test set of my best algorithm")
    #plt.title("PR curve for the test set of the weakened version of my best algorithm")
    plt.show()
