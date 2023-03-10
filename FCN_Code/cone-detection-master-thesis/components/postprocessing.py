import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage.morphology as morph
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm

from utils.image import show_image

def _extended_max(img, h):
    """
    Return local maxima
    """
    mask = img.copy() 
    marker = mask + h  
    hmax =  morph.reconstruction(marker, mask, method='erosion')
    return morph.local_maxima(hmax) 

def _locations_to_mask(loc, size=(144,144)):
    """
    Get a list / numpy array of cone locations
    and create a 2D mask from it
    """
    res = np.zeros(size)
    if len(loc) > 0:
        res[np.round(loc[:,0]).astype(int), np.round(loc[:,1]).astype(int)] = 1
    return res

def naive_postprocessing(prediction, threshold):
    """
    Perform naive postprocessing by simply thresholding
    the prediction, i.e.

    prediction <= threshold
    """
    return np.ones_like(prediction) * (prediction <= threshold)

def hamwoods_postprocessing(prediction, threshold, verbose=False):
    """
    Post-processing of probability map as described in 
    Cunefare et al. (2017)

    with adaptation to DT and using best parameters from lab
    """

    # Config
    sigma = 1.0
    max_h = 0.05

    # Prediction is DT
    prediction = np.max(prediction) - prediction
    threshold = 0#np.max(prediction) - threshold

    # Postprocessing
    filtered = gaussian_filter(prediction, sigma)
    map_max = _extended_max(filtered, max_h)
    # show_image(map_max)
    clusters, num_labels = measure.label(map_max, connectivity=2, return_num=True)

    pos = []
    iterator = tqdm(range(num_labels)) if verbose else range(num_labels)
    for l in iterator:
        indices = np.argwhere(clusters == l)
        y, x = indices[:,0], indices[:,1]
        values = filtered[y,x]

        if np.max(values) > threshold:
            pos.append([np.mean(y), np.mean(x)])

    mask = _locations_to_mask(np.array(pos), size=prediction.shape)
    return mask, pos

def particle_postprocessing(prediction):
    """
    Particle system based post-processing

    TODO: Implement
    """
    return prediction

def evaluate(mask, gt, mode='hamwood',  dist_criterium=2.0, hamwoods_free_zone=False):
    """
    Evaluate a mask of cone locations against
    a ground truth cone locations map, computing 

    - true positives
    - false positives
    - false negatives

    and returning 

    - true positive rate
    - false discovery rate
    - dice coefficient

    using two modes

    - identical: no tolerance (exact location)
    - hamwood: k-NN approach

    Note: Hamwood et al. use a distance of 6.0 - which is way too much
    Let's use either 
    - the Moore neighbourhood (d < 1.5)
    - von Neumann neighbourhood (d < 1 + epsilon)
    - some already large margin (d < 2 + epsilon)
    """
    if mode == 'identical':
        combined = 2 * mask - gt

        # 1 = true positive, 2 = false positive, -1 = false negative, 0 = ignore
        tp = np.sum(combined == 1)
        fp = np.sum(combined == 2)
        fn = np.sum(combined == -1)

    elif mode == 'hamwood':
        tp = 0
        fn = 0
        fp = 0

        border_y, border_x = mask.shape[0] - 3, mask.shape[1] - 3

        gt_indices = np.argwhere(gt == 1)
        pr_indices = np.argwhere(mask == 1)

        chamfer_dist = 0.0

        if gt_indices.shape[0] == 0:
            fp = np.sum(mask == 1)
        elif pr_indices.shape[0] == 0:
            fn = np.sum(gt == 1)
        else: 
            nbrs = NearestNeighbors(n_neighbors=len(pr_indices)).fit(pr_indices)
            distances, indices = nbrs.kneighbors(gt_indices)

            unused_gt = np.ones(len(gt_indices), dtype=np.uint8)
            unused_pred = np.ones(len(pr_indices), dtype=np.uint8)

            # Iterate all gt cones. 
            for gt_idx, pred_indices in enumerate(indices):
                # Iterate predictions by distance
                for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
                    if it_idx == 0:
                        chamfer_dist = chamfer_dist + dist
                    if dist > dist_criterium:
                        break
                    if unused_pred[pred_idx] == 1:
                        unused_pred[pred_idx] = 0
                        unused_gt[gt_idx] = 0
                        tp += 1
                        break  

                # 'Free zone' by Hamwood et al. 
                if hamwoods_free_zone:           
                    if unused_gt[gt_idx] == 1:
                        y, x = gt_indices[gt_idx]
                        if y < 2 or x < 2 or y > border_y or x > border_x:
                            fn -= 1

            # 'Free zone' by Hamwood et al.
            if hamwoods_free_zone:
                for i, pred in enumerate(pr_indices):
                    if unused_pred[i] == 1:
                        y, x = pred
                        if y < 2 or x < 2 or y > border_y or x > border_x:
                            fp -= 1       

            fn += np.sum(unused_gt)
            fp += np.sum(unused_pred)

    # Return true positive rate, false discovery rate and dice coefficient
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    chamfer_dist = chamfer_dist / np.sum(gt == 1)

    return tpr, fdr, dice, chamfer_dist
