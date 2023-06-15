import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage.morphology as morph
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm

from utils.cdc import get_maximum_extent, get_maximum_extent_from_bounding_box, get_circular_mask, arcmin_to_pixels, pixels_to_arcmin
from utils.image import mark_cones, tensor_to_image, mask_array, erode_image, get_distance_mask, class_to_dt
from utils.transforms import ClipTransform, ErodeTransform

def _extended_max(img, h):
    """
    Return local maxima
    """
    mask = img.copy() 
    marker = mask + h  
    hmax =  morph.reconstruction(marker, mask, method='erosion')
    return morph.local_maxima(hmax, allow_borders=True) 

def min_pp(img, h):
    """
    Return local minima
    """
    mask = img.copy() 
    marker = mask + h
    #hmax =  morph.reconstruction(marker, mask, method='erosion')
    hmax = morph.reconstruction(mask, marker, method='dilation')
    return morph.local_minima(hmax, allow_borders=True)

def coordinates_to_mask(loc, size=(144,144)):
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

def minimum_postprocessing(prediction, verbose=False, return_coordinates=False):
    """
    Modified version of post-processing
    which simply extracts local minima after some
    morphing
    """
    # Config
    max_h = 0.25

    # Postprocessing
    map_min = min_pp(prediction, max_h)
    clusters, num_labels = measure.label(map_min, connectivity=2, return_num=True)

    pos = []
    iterator = tqdm(range(1,num_labels+1)) if verbose else range(1,num_labels+1)
    for l in iterator:
        indices = np.argwhere(clusters == l)
        y, x = indices[:,0], indices[:,1]
        pos.append([np.mean(y), np.mean(x)])

    if return_coordinates:
        return np.array(pos)
    else:
        mask = coordinates_to_mask(np.array(pos), size=prediction.shape)
        return mask

def hamwoods_postprocessing(prediction, threshold, verbose=False, is_dt=True, prefilter=True, return_coordinates=False):
    """
    Post-processing of probability map as described in 
    Cunefare et al. (2017)

    with adaptation to DT and using best parameters from lab
    """

    # Config
    sigma = 1.0
    max_h = 0.05
    cut_off = 5

    # Prediction is DT
    if is_dt:
        prediction[prediction > cut_off] = cut_off
        prediction = np.max(prediction) - prediction
        threshold = np.max(prediction) - threshold

    # Postprocessing
    if not prefilter:
        filtered = prediction
    else:
        filtered = gaussian_filter(prediction, sigma)

    map_max = _extended_max(filtered, max_h)
    clusters, num_labels = measure.label(map_max, connectivity=2, return_num=True)

    pos = []
    iterator = tqdm(range(1,num_labels+1)) if verbose else range(1,num_labels+1)
    for l in iterator:
        indices = np.argwhere(clusters == l)
        y, x = indices[:,0], indices[:,1]
        values = filtered[y,x]

        if np.max(values) > threshold:
            pos.append([np.mean(y), np.mean(x)])

    if return_coordinates:
        return np.array(pos)
    else:
        mask = coordinates_to_mask(np.array(pos), size=prediction.shape)
        return mask

@DeprecationWarning
def particle_postprocessing(prediction):
    """
    USE utils/particles.py

    Particle system based post-processing
    """
    return prediction

# def get_distance_to_closest_neighbour(cones):
#     """
#     Get the distance to the closest neighbour
#     """
#     cones = cones.astype(float)
#     pr_indices = np.argwhere(cones == 1).astype(int)
#     distance_map = np.zeros_like(cones)

#     if pr_indices.shape[0] > 0:
#         # We only need the closest neighbor
#         n_neighbors = min(2, len(pr_indices))

#         nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
#         distances, _ = nbrs.kneighbors(pr_indices)

#         for idx, dist in enumerate(distances):
#             y, x = pr_indices[idx]
#             distance_map[y,x] = dist[1] # 0 = itself, 1 = nearest neighbour
        
#     return distance_map

# def get_average_distance_to_n_neighbours(cones, n):
#     """
#     Get the average distance to n neighbours
#     """
#     cones = cones.astype(float)
#     pr_indices = np.argwhere(cones == 1).astype(int)
#     distance_map = np.zeros_like(cones)

#     if pr_indices.shape[0] > 0:
#         # We only need the closest neighbor
#         n_neighbors = min(n + 1, len(pr_indices))

#         nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
#         distances, _ = nbrs.kneighbors(pr_indices)

#         for idx, dist in enumerate(distances):
#             y, x = pr_indices[idx]
#             distance_map[y,x] = np.mean(dist[1:]) # 0 = itself, 1 = nearest neighbour
        
#     return distance_map

# def get_mean_neighbour_distance_over_eccentricity(cones, cy, cx, max_eccentricity=90):
#     """
#     Get the mean distance to the neighbouring cone (GT or prediction)
#     over eccentricity
#     """
#     cones = cones.astype(float)
#     pr_indices = np.argwhere(cones == 1).astype(int)
#     distance_map = np.zeros_like(cones)
#     mean_distances = np.zeros(max_eccentricity)

#     if pr_indices.shape[0] == 0:
#         return mean_distances # distance_map
#     else: 
#         # We only need the closest neighbor
#         n_neighbors = min(2, len(pr_indices))

#         nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
#         distances, _ = nbrs.kneighbors(pr_indices)

#         for idx, dist in enumerate(distances):
#             y, x = pr_indices[idx]
#             distance_map[y,x] = dist[1] # 0 = itself, 1 = nearest neighbour

#         # return distance_map
#         pixels_per_degree = 600 # 600 is specific to UKB dataset

#         extent = get_maximum_extent(cones.shape[0], cones.shape[1], cy, cx, pixels_per_degree) 

#         extent = int(np.ceil(extent)) # Ceil to next integer extent
#         for eccentricity in range(extent):
#             extent_mask = get_circular_mask(cones.shape[0], cones.shape[1], cy, cx, 
#                 arcmin_to_pixels(pixels_per_degree, eccentricity), arcmin_to_pixels(pixels_per_degree, eccentricity + 1))

#             subset = distance_map * extent_mask
#             count = np.sum(subset > 0.0)
#             if count > 0:
#                 mean_distance = np.sum(subset) / count
#                 mean_distances[eccentricity] = mean_distance
        
#         return mean_distances

# def evaluate(mask, gt, mode='hamwood',  dist_criterium=2.0, hamwoods_free_zone=False, dist_criterium_map=None):
#     """
#     Evaluate a mask of cone locations against
#     a ground truth cone locations map, computing 

#     - true positives
#     - false positives
#     - false negatives

#     and returning 

#     - true positive rate
#     - false discovery rate
#     - dice coefficient

#     using two modes

#     - identical: no tolerance (exact location)
#     - hamwood: k-NN approach

#     Note: Hamwood et al. use a distance of 6.0 - which is way too much
#     Let's use either 
#     - the Moore neighbourhood (d < 1.5)
#     - von Neumann neighbourhood (d < 1 + epsilon)
#     - some already large margin (d < 2 + epsilon)
#     """
#     if mode == 'identical':
#         combined = 2 * mask - gt

#         # 1 = true positive, 2 = false positive, -1 = false negative, 0 = ignore
#         tp = np.sum(combined == 1)
#         fp = np.sum(combined == 2)
#         fn = np.sum(combined == -1)

#     elif mode == 'hamwood':
#         tp = 0
#         fn = 0
#         fp = 0

#         border_y, border_x = mask.shape[0] - 3, mask.shape[1] - 3

#         gt_indices = np.argwhere(gt == 1).astype(int)
#         pr_indices = np.argwhere(mask == 1).astype(int)

#         chamfer_dist = 0.0

#         if gt_indices.shape[0] == 0:
#             fp = np.sum(mask == 1)
#         elif pr_indices.shape[0] == 0:
#             fn = np.sum(gt == 1)
#         else: 
#             # len(pr_indices) is computationally infeasible for larger images
#             # so let's use a lenient upper bound of GT cones around the 
#             # predicted cone
#             n_neighbors = min(128, len(pr_indices))

#             nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
#             distances, indices = nbrs.kneighbors(gt_indices)

#             unused_gt = np.ones(len(gt_indices), dtype=np.uint8)
#             unused_pred = np.ones(len(pr_indices), dtype=np.uint8)

#             # Iterate all gt cones. 
#             for gt_idx, pred_indices in enumerate(indices):
#                 # Get custom dist_criterium if dist_criterium_map is available
#                 if dist_criterium_map is not None:
#                     y, x = gt_indices[gt_idx]
#                     dist_criterium = dist_criterium_map[y,x]

#                 # Iterate predictions by distance
#                 for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
#                     if it_idx == 0:
#                         chamfer_dist = chamfer_dist + dist
#                     if dist > dist_criterium:
#                         break
#                     if unused_pred[pred_idx] == 1:
#                         unused_pred[pred_idx] = 0
#                         unused_gt[gt_idx] = 0
#                         tp += 1
#                         break  

#                 # 'Free zone' by Hamwood et al. 
#                 if hamwoods_free_zone:           
#                     if unused_gt[gt_idx] == 1:
#                         y, x = gt_indices[gt_idx]
#                         if y < 2 or x < 2 or y > border_y or x > border_x:
#                             fn -= 1

#             # 'Free zone' by Hamwood et al.
#             if hamwoods_free_zone:
#                 for i, pred in enumerate(pr_indices):
#                     if unused_pred[i] == 1:
#                         y, x = pred
#                         if y < 2 or x < 2 or y > border_y or x > border_x:
#                             fp -= 1       

#             fn += np.sum(unused_gt)
#             fp += np.sum(unused_pred)

#     # Return true positive rate, false discovery rate and dice coefficient
#     tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
#     fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
#     dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
#     if np.sum(gt == 1) > 0:
#         chamfer_dist = chamfer_dist / np.sum(gt == 1)

#     #print(tpr, fdr, dice, chamfer_dist)
#     return tpr, fdr, dice, chamfer_dist

def evaluate(tp, fp, fn, chamfer):
    """
    Compute TPR, FDR, Dice score and Chamfer distance
    """
    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()

    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    
    if tp + fn > 0:
        chamfer = chamfer.sum() / (tp + fn)

    return tpr, fdr, dice, chamfer

@DeprecationWarning
def map_gt_to_exact(label, exact_locations):
    """
    Map a label to exact annotated cone location
    """
    n_neighbors = 128

    gt_indices = np.argwhere(label == 1)
    pr_indices = exact_locations

    result = np.zeros_like(gt_indices).astype(float)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
    distances, indices = nbrs.kneighbors(gt_indices)

    # Iterate all gt cones. 
    for gt_idx, pred_indices in enumerate(indices):       
        # Iterate predictions by distance
        for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
            if it_idx == 0:
                result[gt_idx,:] = pr_indices[pred_idx,:]

    return result


def get_chamfer_distances_exact(shape, gt_indices, pr_indices):
    """
    Get exact chamfer distances
    """
    #chamfer = 0.0 #np.zeros(shape)
    chamfer = np.zeros(len(gt_indices))
    n_neighbors = 128 #min(128, len(pr_indices))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
    distances, indices = nbrs.kneighbors(gt_indices)

    # Iterate all gt cones. 
    for gt_idx, pred_indices in enumerate(indices):       
        # Iterate predictions by distance
        for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
            if it_idx == 0:
                chamfer[gt_idx] = dist
                #chamfer += dist
                # y, x = pr_indices[pred_idx]
                # chamfer[y,x] = dist

    return chamfer #/ indices.shape[0] # Mean chamfer

def get_tp_fp_fn_chamfer(mask, gt, dist_criterium=2.0, dist_criterium_map=None, free_border=False, image=None, exact_locs=None):
    """
    Get a map of TP, FP, FN and Chamfer Distance using Hamwood's 
    post-processing without a free zone
    """
    tp = np.zeros_like(mask)
    fn = np.zeros_like(mask)
    fp = np.zeros_like(mask)
    chamfer = np.zeros_like(mask)

    gt_indices = np.argwhere(gt == 1).astype(int)
    pr_indices = np.argwhere(mask == 1).astype(int)

    if exact_locs is not None:
        gt_indices = map_gt_to_exact(gt, exact_locs)

    if gt_indices.shape[0] == 0:
        fp = mask
    elif pr_indices.shape[0] == 0:
        fn = gt
    else: 
        # len(pr_indices) is computationally infeasible for larger images
        # so let's use a lenient upper bound of GT cones around the 
        # predicted cone
        n_neighbors = min(128, len(pr_indices))

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pr_indices)
        distances, indices = nbrs.kneighbors(gt_indices)

        unused_gt = np.ones(len(gt_indices), dtype=np.uint8)
        unused_pred = np.ones(len(pr_indices), dtype=np.uint8)

        # Iterate all gt cones. 
        for gt_idx, pred_indices in enumerate(indices):
            # Get custom dist_criterium if dist_criterium_map is available
            if dist_criterium_map is not None:
                y, x = gt_indices[gt_idx]
                y, x = int(np.round(y)), int(np.round(x))
                dist_criterium = dist_criterium_map[y,x]
            
            # Iterate predictions by distance
            for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
                if it_idx == 0:
                    y, x = pr_indices[pred_idx]
                    chamfer[y,x] = dist
                    #chamfer_dist = chamfer_dist + dist
                if dist > dist_criterium:
                    break
                if unused_pred[pred_idx] == 1:
                    unused_pred[pred_idx] = 0
                    unused_gt[gt_idx] = 0
                    break  

        # False negatives
        for idx, is_unused in enumerate(unused_gt): 
            if is_unused == 1:
                y, x = gt_indices[idx]   
                y, x = int(np.round(y)), int(np.round(x))
                fn[y,x] = 1

        # False and true positives
        for idx, is_unused in enumerate(unused_pred):
            y, x = pr_indices[idx]
            if is_unused == 1:             
                fp[y,x] = 1
            else:
                tp[y,x] = 1

    if free_border:
        eroded, _ = erode_image(image, iterations=4)
        tp = mask_array(tp, eroded, True)
        fp = mask_array(fp, eroded, True)
        fn = mask_array(fn, eroded, True)
        chamfer = mask_array(chamfer, eroded, True)

    return tp, fp, fn, chamfer

@DeprecationWarning
def remove_some_fn(mask, image, cy, cx, mean_distances, epsilon=1.25):
    """
    Find regions in the mask where too few cones are located 
    w.r.t. to the mean cone distances (over eccentricity)
    """
    mask_dt = mask_array(class_to_dt(mask), image, better_mask=True)
    dmask_mean_dst = get_distance_mask(image.shape, cy, cx, 1.5, fit_to=mean_distances)
    dmask_mean_dst[dmask_mean_dst == 0] = np.max(dmask_mean_dst)
    uninteresting = mask_dt <= epsilon * dmask_mean_dst
    add_mask = minimum_postprocessing(uninteresting.astype(float), verbose=False)
    return mask + add_mask

def compute_stats(model, free_border=False, save_data=True,
    use_cdc=True, dataset="test", max_eccentricity=90, initial_dist=1.5):
    """
    Test this instance of a FCN with the given
    parameters
    """
    clip = ClipTransform()
    erode = ErodeTransform()

    used_dataloader = model.test_dataloader
    if dataset == "train":
        used_dataloader = model.train_dataloader
    if dataset == "validation":
        used_dataloader = model.validation_dataloader

    # Accumulators and post-processing constants
    dataset_num = len(used_dataloader.dataset)

    tprs = np.zeros((dataset_num,max_eccentricity))
    fdrs = np.zeros((dataset_num,max_eccentricity))
    dices = np.zeros((dataset_num,max_eccentricity))
    chamfers = np.zeros((dataset_num,max_eccentricity))
    gt_cones = np.zeros((dataset_num,max_eccentricity))
    gt_density = np.zeros((dataset_num,max_eccentricity))
    pred_density = np.zeros((dataset_num,max_eccentricity))
    gt_mean_distance = np.zeros((dataset_num,max_eccentricity))
    pred_mean_distance = np.zeros((dataset_num,max_eccentricity))
    
    wi_tprs, wi_fdrs, wi_dices, wi_chamfers = np.zeros(dataset_num), np.zeros(dataset_num), np.zeros(dataset_num), np.zeros(dataset_num)

    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(used_dataloader)):
            imgs, labels, ids, cdc, cdc_distances = data

            imgs = clip(imgs)
            labels = clip(labels)

            # Erode image / label
            # imgs, applied = erode(imgs, labels)
            # labels = applied[0]

            # Feed forward
            image = model.prepare_data(imgs)
            pred = model(image)[0,0,:,:].detach().cpu()
            image = image.detach().cpu()

            # Clip regions outside of ROI
            pred = tensor_to_image(pred)
            image = tensor_to_image(image)
            gt = tensor_to_image(labels)

            # Post-process
            pred = mask_array(pred, image, better_mask=True, fill=1) # 0 in DT denotes cone!
            #proc = hamwoods_postprocessing(pred, threshold, verbose=True)
            proc = minimum_postprocessing(pred, verbose=True)
            pred = mask_array(pred, image, better_mask=True, fill=0) # Undo masking with 1

            # Remove regions around image montage for evaluation
            proc = mask_array(proc, image, better_mask=True, fill=0)
            gt = (mask_array(gt, image, better_mask=True, fill=1) == 0) # 0 in DT denotes cone!

            # Get CDC (or approximate CDC)
            if use_cdc:
                cy, cx = cdc[0][0] # array of arrays in .npz file
                cy, cx = cy.item(), cx.item() # cdc is Tensor
            else:
                # Estimate CDC by mean and get distance mask
                estimated_cdc = np.mean(np.argwhere(gt == 1), axis=0)
                cy, cx = estimated_cdc

            # Get distance mask for evaluation
            dmask = get_distance_mask(image.shape, cy, cx, initial_dist) # Should not be smaller than sqrt(2) to allow diagonal offsets
            
            # Compute TPR, FDR, Dice and Chamfer for regions around CDC
            pixels_per_degree = 600 # 600 is specific to UKB dataset

            # Get TP, FP, FN masks
            tp, fp, fn, chamfer = get_tp_fp_fn_chamfer(proc, gt, dist_criterium_map=dmask, free_border=free_border, image=image)

            extent = get_maximum_extent(gt.shape[0], gt.shape[1], cy, cx, pixels_per_degree) 
            #extent = get_maximum_extent_from_bounding_box(image, cy, cx, pixels_per_degree)
            #extent = np.min([extent, 15]) # Cut-off
            extent = int(np.ceil(extent)) # Ceil to next integer extent

            for eccentricity in range(extent):
                extent_mask = get_circular_mask(gt.shape[0], gt.shape[1], cy, cx, 
                    arcmin_to_pixels(pixels_per_degree, eccentricity), arcmin_to_pixels(pixels_per_degree, eccentricity + 1))

                # proc_extent = proc * extent_mask
                # gt_extent = gt * extent_mask
                
                # #tpr, fdr, dice, chamfer = evaluate(proc_extent, gt_extent, dist_criterium=dist, hamwoods_free_zone=free_border)
                # tpr, fdr, dice, chamfer = evaluate(proc_extent, gt_extent, dist_criterium_map=dmask, hamwoods_free_zone=free_border)

                tp_extent = tp * extent_mask
                fp_extent = fp * extent_mask
                fn_extent = fn * extent_mask
                chamfer_extent = chamfer * extent_mask

                gt_extent = gt * extent_mask
                proc_extent = tp_extent + fp_extent

                tpr, fdr, dice, chamfer_dist = evaluate(tp_extent, fp_extent, fn_extent, chamfer_extent)

                gt_log = np.sum(gt_extent)
                pred_log = np.sum(proc_extent)
                norm = np.pi * (2.0 * eccentricity + 1)
                gt_log = gt_log / norm * 60 ** 2 # arcmin^2 to deg^2
                pred_log = pred_log / norm * 60 ** 2 #arcmin^2 to deg^2

                tprs[idx,eccentricity] = tpr
                fdrs[idx,eccentricity] = fdr
                dices[idx,eccentricity] = dice
                chamfers[idx,eccentricity] = chamfer_dist

                gt_cones[idx,eccentricity] = np.sum(gt_extent)
                gt_density[idx,eccentricity] = gt_log
                pred_density[idx,eccentricity] = pred_log  

            # Compute distance to closest neighbour
            gt_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(gt, cy, cx)
            pred_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(proc, cy, cx)

            # Whole image
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium=dist, hamwoods_free_zone=free_border)
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium_map=dmask, hamwoods_free_zone=free_border)
            tpr, fdr, dice, chamfer_dist = evaluate(tp, fp, fn, chamfer)

            wi_tprs[idx] = tpr
            wi_fdrs[idx] = fdr
            wi_dices[idx] = dice
            wi_chamfers[idx] = chamfer_dist

            # Optionally save data
            if save_data:
                # Convert arrays to save disk storage
                np.savez(f"../stats/{ids[0]}.npz", 
                    image=image.astype(float), 
                    label=gt.astype(np.uint8),
                    prediction=pred.astype(float),
                    postprocessed=proc.astype(np.uint8),
                    tp=tp.astype(np.uint8),
                    fp=fp.astype(np.uint8),
                    fn=fn.astype(np.uint8),
                    chamfer=chamfer.astype(float))

            # Clean up manually
            del imgs, labels, ids, cdc, pred, proc, gt #, applied
            torch.cuda.empty_cache()

    model.train()
    
    # Save stats
    np.savez("../stats/stats_cdc.npz", tprs=tprs, fdrs=fdrs, dices=dices, chamfers=chamfers, gt_cones=gt_cones, 
        gt_density=gt_density, pred_density=pred_density, gt_mean_distance=gt_mean_distance, pred_mean_distance=pred_mean_distance)
    np.savez("../stats/stats_whole_image.npz", wi_tprs=wi_tprs, wi_fdrs=wi_fdrs, wi_dices=wi_dices, wi_chamfers=wi_chamfers)
