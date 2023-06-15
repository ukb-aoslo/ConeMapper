import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage.morphology as morph
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm
from scipy.io import loadmat
import os

from utils.cdc import get_maximum_extent, get_maximum_extent_from_bounding_box, get_circular_mask, arcmin_to_pixels, pixels_to_arcmin
from utils.image import mark_cones, tensor_to_image, mask_array, erode_image, get_distance_mask, class_to_dt
from utils.transforms import ClipTransform, ErodeTransform

def extended_max(img, h):
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

def coordinates_to_mask(loc, size):
    """
    Get a list / numpy array of cone locations
    and create a 2D mask from it
    """
    res = np.zeros(size)
    if len(loc) > 0:
        res[np.round(loc[:,0]).astype(int), np.round(loc[:,1]).astype(int)] = 1
    return res

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

    map_max = extended_max(filtered, max_h)
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
    
def evaluate_grid(tp, fp, fn, chamfer):
    """
    Compute TPR, FDR, Dice score and Chamfer distance
    from a grid of TP, FP, FN and chamfer distances
    """
    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()

    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    
    if tp + fn > 0:
        chamfer = chamfer.sum() / (tp + fn)

    return tpr, fdr, dice, chamfer

def evaluate_coordinates(tp, fp, fn, chamfer):
    """
    Compute TPR, FDR, Dice score and Chamfer distance
    from a list of coordinates for TP, FP, FN and chamfer 
    distances
    """
    tp, fp, fn = len(tp), len(fp), len(fn)

    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    
    if tp + fn > 0:
        chamfer_dist = chamfer.sum() / (tp + fn)
    else:
        chamfer_dist = 0

    return tpr, fdr, dice, chamfer_dist

def get_tp_fp_fn_chamfer_grid(mask, gt, dist_criterium=2.0, dist_criterium_map=None, free_border=False, image=None):
    """
    Get a map of TP, FP, FN and Chamfer Distance using Minimum Postprocessing
    optionally with a free zone from a grid of cones and GT
    """
    tp = np.zeros_like(mask)
    fn = np.zeros_like(mask)
    fp = np.zeros_like(mask)
    chamfer = np.zeros_like(mask)

    # Check for cones inside image area
    # if free_border:
    #     image, _ = erode_image(image, iterations=4) # Free border of 4 pixels at the border of the montage
    # boundary_mask = (mask_array(image, image, True, fill=32) != 32)

    gt_indices = np.argwhere(gt == 1).astype(int)
    pr_indices = np.argwhere(mask == 1).astype(int)

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
            # Out of bounds check
            # y, x = gt_indices[gt_idx]
            # if not boundary_mask[int(y), int(x)]:
            #     continue

            # Get custom dist_criterium if dist_criterium_map is available
            if dist_criterium_map is not None:
                y, x = gt_indices[gt_idx]
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

def get_tp_fp_fn_chamfer_coordinates(cones, gt_cones, image, dist_criterium=2.0, dist_criterium_map=None, free_border=False):
    """
    Get a list of TP, FP, FN and Chamfer Distance using Minimum Postprocessing
    optionally with a free zone from a grid of cones and GT
    """
    tp = []
    fn = []
    fp = []
    chamfer = []

    # Check for cones inside image area
    if free_border:
        image, _ = erode_image(image, iterations=4) # Free border of 4 pixels at the border of the montage
    boundary_mask = (mask_array(image, image, True, fill=32) != 32)

    #gt_indices = gt_cones 
    #pr_indices = cones 

    if len(gt_cones) == 0:
        fp = cones
    elif len(cones) == 0:
        fn = gt_cones
    else: 
        # len(pr_indices) is computationally infeasible for larger images
        # so let's use a lenient upper bound of GT cones around the 
        # predicted cone
        n_neighbors = min(128, len(cones))

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(cones)
        distances, indices = nbrs.kneighbors(gt_cones)

        unused_gt = np.ones(len(gt_cones), dtype=np.uint8)
        unused_pred = np.ones(len(cones), dtype=np.uint8)

        # Iterate all gt cones. 
        for gt_idx, pred_indices in enumerate(indices):
            # Out of bounds check
            y, x = gt_cones[gt_idx]
            if not boundary_mask[int(y), int(x)]:
                continue

            # Get custom dist_criterium if dist_criterium_map is available
            if dist_criterium_map is not None:
                y, x = gt_cones[gt_idx]
                y, x = int(np.round(y)), int(np.round(x))
                dist_criterium = dist_criterium_map[y, x]
            
            # Iterate predictions by distance
            #print(dist_criterium, distances[gt_idx,:8])
            for it_idx, (pred_idx, dist) in enumerate(zip(pred_indices, distances[gt_idx])):
                if it_idx == 0:
                    y, x = cones[pred_idx]
                    if boundary_mask[int(y), int(x)]:
                        chamfer.append([y,x,dist])
                        #chamfer[y,x] = dist
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
                y, x = gt_cones[idx]        
                if boundary_mask[int(y), int(x)]:
                    #fn[y,x] = 1
                    fn.append([y,x])

        # False and true positives
        for idx, is_unused in enumerate(unused_pred):
            y, x = cones[idx]
            if is_unused == 1:  
                if boundary_mask[int(y), int(x)]:           
                    #fp[y,x] = 1
                    fp.append([y,x])
            else:
                if boundary_mask[int(y), int(x)]:
                    #tp[y,x] = 1
                    tp.append([y,x])

    return np.array(tp), np.array(fp), np.array(fn), np.array(chamfer)

def evaluate_distances_by_eccentricity_grid(cones, cy, cx, n_neighbors=3, pixels_per_degree=600, max_eccentricity=90):
    """
    Same as 'evaluate_distances_by_eccentricity_coordinates' but 
    with cones being a 2D map of cones
    """
    locs = np.argwhere(cones == 1)
    return evaluate_distances_by_eccentricity_coordinates(locs, cy, cx, n_neighbors=n_neighbors, pixels_per_degree=pixels_per_degree, max_eccentricity=max_eccentricity)

def evaluate_distances_by_eccentricity_coordinates(cones, cy, cx, n_neighbors=3, pixels_per_degree=600, max_eccentricity=90):
    """
    a) Compute for each cone the distances to its nearest neighbors

    b) Accumulate mean distance over eccentricity (convert distance to CDC from pixels to arcmin)

    c) Return resulting mean distance (and std) for each eccentricity
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cones) # No particle is a neighbor to itself, hence + 1
    distances, indices = nbrs.kneighbors(cones)
    distances, indices = distances[:,1:], indices[:,1:] # No particle is a neighbor to itself

    mean_distances = distances.mean(axis=1)

    distances_cdc = np.sqrt(np.sum((cones - np.array([cy,cx])) ** 2, axis=1))
    eccentricities = distances_cdc / pixels_per_degree * 60.0

    result = [[] for _ in range(max_eccentricity)]

    for (eccentricity, mean_distance) in zip(eccentricities, mean_distances):
        idx = int(eccentricity)
        if idx < max_eccentricity:
            result[idx].append(mean_distance)
    
    result_mean = np.array([np.array(arr).mean() for arr in result])
    result_std = np.array([np.array(arr).std() for arr in result])

    return result_mean, result_std

@DeprecationWarning
def compute_stats_grid(model, free_border=False, save_data=True,
    use_cdc=True, dataset="test", max_eccentricity=90, initial_dist=1.5):
    """
    Test this instance of a FCN with the given
    parameters
    """
    clip = ClipTransform()

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
            imgs, labels, ids, cdc, cdc_distances, exact_gt = data

            imgs = clip(imgs)
            labels = clip(labels)

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
                #cy, cx = cdc[0][0] # array of arrays in .npz file
                cx, cy = cdc[0][0] # IMPORTANT: CDC is in order (x,y) + array of arrays in .npz file
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
            tp, fp, fn, chamfer = get_tp_fp_fn_chamfer_grid(proc, gt, dist_criterium_map=dmask, free_border=free_border, image=image)

            #extent = get_maximum_extent(gt.shape[0], gt.shape[1], cy, cx, pixels_per_degree) 
            extent = get_maximum_extent_from_bounding_box(image, cy, cx, pixels_per_degree)
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

                tpr, fdr, dice, chamfer_dist = evaluate_grid(tp_extent, fp_extent, fn_extent, chamfer_extent)

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

            # Compute local mean distances
            gt_mean_distance[idx], _ = evaluate_distances_by_eccentricity_grid(gt, cy, cx)
            pred_mean_distance[idx], _ = evaluate_distances_by_eccentricity_grid(proc, cy, cx)
            # gt_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(gt, cy, cx)
            # pred_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(proc, cy, cx)

            # Whole image
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium=dist, hamwoods_free_zone=free_border)
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium_map=dmask, hamwoods_free_zone=free_border)
            tpr, fdr, dice, chamfer_dist = evaluate_grid(tp, fp, fn, chamfer)

            wi_tprs[idx] = tpr
            wi_fdrs[idx] = fdr
            wi_dices[idx] = dice
            wi_chamfers[idx] = chamfer_dist

            # Optionally save data
            if save_data:
                # Convert arrays to save disk storage
                np.savez(f"../stats/grid/{ids[0]}.npz", 
                    image=image.astype(float), 
                    label=gt.astype(np.uint8),
                    prediction=pred.astype(float),
                    postprocessed=proc.astype(np.uint8),
                    tp=tp.astype(np.uint8),
                    fp=fp.astype(np.uint8),
                    fn=fn.astype(np.uint8),
                    chamfer=chamfer.astype(float))

            # Clean up manually
            del imgs, labels, ids, cdc, pred, proc, gt, exact_gt #, applied
            torch.cuda.empty_cache()

    model.train()
    
    # Save stats
    np.savez("../stats/grid/stats_cdc.npz", tprs=tprs, fdrs=fdrs, dices=dices, chamfers=chamfers, gt_cones=gt_cones, 
        gt_density=gt_density, pred_density=pred_density, gt_mean_distance=gt_mean_distance, pred_mean_distance=pred_mean_distance)
    np.savez("../stats/grid/stats_whole_image.npz", wi_tprs=wi_tprs, wi_fdrs=wi_fdrs, wi_dices=wi_dices, wi_chamfers=wi_chamfers)

def compute_stats_coordinates(model, free_border=False, save_data=True,
    use_cdc=True, dataset="test", max_eccentricity=90, initial_dist=1.5,
    identifier=None):
    """
    Test this instance of a FCN with the given
    parameters
    """
    clip = ClipTransform()

    path = "coordinates"

    if identifier is not None:
        path = f"{identifier}/{path}"
        os.makedirs(f"../stats/{path}", exist_ok=True)

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
            imgs, labels, ids, cdc, cdc_distances, exact_gt = data

            imgs = clip(imgs)
            labels = clip(labels)

            # Feed forward
            image = model.prepare_data(imgs)
            pred = model(image)[0,0,:,:].detach().cpu()
            image = image.detach().cpu()

            # Clip regions outside of ROI
            pred = tensor_to_image(pred)
            image = tensor_to_image(image)
            gt = tensor_to_image(labels)
            exact_gt = tensor_to_image(exact_gt)

            # Post-process
            pred = mask_array(pred, image, better_mask=True, fill=1) # 0 in DT denotes cone!
            #proc = hamwoods_postprocessing(pred, threshold, verbose=True)
            proc = minimum_postprocessing(pred, verbose=True, return_coordinates=True)
            pred = mask_array(pred, image, better_mask=True, fill=0) # Undo masking with 1

            # Remove regions around image montage for evaluation
            #proc = mask_array(proc, image, better_mask=True, fill=0)
            gt = (mask_array(gt, image, better_mask=True, fill=1) == 0) # 0 in DT denotes cone!

            # Get CDC (or approximate CDC)
            if use_cdc:
                #cy, cx = cdc[0][0] # array of arrays in .npz file
                cx, cy = cdc[0][0] # IMPORTANT: CDC is in order (x,y) + array of arrays in .npz file
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
            tp, fp, fn, chamfer = get_tp_fp_fn_chamfer_coordinates(proc, exact_gt, image, dist_criterium_map=dmask, free_border=free_border)

            #extent = get_maximum_extent(gt.shape[0], gt.shape[1], cy, cx, pixels_per_degree) 
            extent = get_maximum_extent_from_bounding_box(image, cy, cx, pixels_per_degree)
            #extent = np.min([extent, 15]) # Cut-off
            extent = int(np.ceil(extent)) # Ceil to next integer extent#

            gt_mask = np.sqrt((exact_gt[:,0] - cy) ** 2 + (exact_gt[:,1] - cx) ** 2)
            tp_mask = np.sqrt((tp[:,0] - cy) ** 2 + (tp[:,1] - cx) ** 2)
            fp_mask = np.sqrt((fp[:,0] - cy) ** 2 + (fp[:,1] - cx) ** 2)
            fn_mask = np.sqrt((fn[:,0] - cy) ** 2 + (fn[:,1] - cx) ** 2)
            ch_mask = np.sqrt((chamfer[:,0] - cy) ** 2 + (chamfer[:,1] - cx) ** 2)

            for eccentricity in range(extent):
                low, high = arcmin_to_pixels(pixels_per_degree, eccentricity), arcmin_to_pixels(pixels_per_degree, eccentricity + 1)
                gt_ecc = np.argwhere(np.logical_and(gt_mask >= low, gt_mask <= high))
                tp_ecc = np.argwhere(np.logical_and(tp_mask >= low, tp_mask <= high))
                fp_ecc = np.argwhere(np.logical_and(fp_mask >= low, fp_mask <= high))
                fn_ecc = np.argwhere(np.logical_and(fn_mask >= low, fn_mask <= high))
                ch_ecc = np.argwhere(np.logical_and(ch_mask >= low, ch_mask <= high))

                gt_extent = exact_gt[gt_ecc]
                tp_extent = tp[tp_ecc]
                fp_extent = fp[fp_ecc]
                fn_extent = fn[fn_ecc]
                chamfer_extent = chamfer[ch_ecc,2] # Chamfer contains (y,x,chamfer_dist)!
                #print(len(gt_extent), len(tp_extent), len(fp_extent), len(fn_extent), len(chamfer_extent))

                tpr, fdr, dice, chamfer_dist = evaluate_coordinates(tp_extent, fp_extent, fn_extent, chamfer_extent)

                gt_log = len(gt_extent)
                pred_log = len(tp_extent) + len(fn_extent)
                norm = np.pi * (2.0 * eccentricity + 1)
                gt_log = gt_log / norm * 60 ** 2 # arcmin^2 to deg^2
                pred_log = pred_log / norm * 60 ** 2 #arcmin^2 to deg^2

                tprs[idx,eccentricity] = tpr
                fdrs[idx,eccentricity] = fdr
                dices[idx,eccentricity] = dice
                chamfers[idx,eccentricity] = chamfer_dist

                gt_cones[idx,eccentricity] = len(gt_extent)
                gt_density[idx,eccentricity] = gt_log
                pred_density[idx,eccentricity] = pred_log  

            # Compute local mean distances
            gt_mean_distance[idx], _ = evaluate_distances_by_eccentricity_coordinates(exact_gt, cy, cx)
            pred_mean_distance[idx], _ = evaluate_distances_by_eccentricity_coordinates(proc, cy, cx)
            # gt_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(gt, cy, cx)
            # pred_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(proc, cy, cx)

            # Whole image
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium=dist, hamwoods_free_zone=free_border)
            # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium_map=dmask, hamwoods_free_zone=free_border)
            tpr, fdr, dice, chamfer_dist = evaluate_coordinates(tp, fp, fn, chamfer[:,2]) # Chamfer contains (y,x,chamfer_dist)!

            wi_tprs[idx] = tpr
            wi_fdrs[idx] = fdr
            wi_dices[idx] = dice
            wi_chamfers[idx] = chamfer_dist

            # Optionally save data
            if save_data:
                # Convert arrays to save disk storage
                np.savez(f"../stats/{path}/{ids[0]}.npz", 
                    image=image.astype(float), 
                    label=gt.astype(np.uint8),
                    prediction=pred.astype(float),
                    postprocessed=proc.astype(float), #.astype(np.uint8),
                    tp=tp.astype(float), # .astype(np.uint8),
                    fp=fp.astype(float), #.astype(np.uint8),
                    fn=fn.astype(float), #.astype(np.uint8),
                    exact_gt=exact_gt.astype(float),
                    chamfer=chamfer.astype(float))

            # Clean up manually
            del imgs, labels, ids, cdc, pred, proc, gt, exact_gt #, applied
            torch.cuda.empty_cache()

    model.train()
    
    # Save stats
    np.savez(f"../stats/{path}/stats_cdc.npz", tprs=tprs, fdrs=fdrs, dices=dices, chamfers=chamfers, gt_cones=gt_cones, 
        gt_density=gt_density, pred_density=pred_density, gt_mean_distance=gt_mean_distance, pred_mean_distance=pred_mean_distance)
    np.savez(f"../stats/{path}/stats_whole_image.npz", wi_tprs=wi_tprs, wi_fdrs=wi_fdrs, wi_dices=wi_dices, wi_chamfers=wi_chamfers)

def compute_stats_hamwood_reference(free_border=False, save_data=True,
    use_cdc=True, max_eccentricity=90, initial_dist=1.5):
    """
    Test this instance of a FCN with the given
    parameters
    """
    #clip = ClipTransform()

    # Path of files
    path = "F:/Data/ConeDatasets/20230301_cross_eroded/hamwood/test"
    labels = ["BAK1012L", "BAK1021R", "BAK1034R", "BAK1040L", "BAK1040R", "BAK1041L", "BAK1064R", "BAK1086R", "BAK1090R", "BAK8015L"]

    # Accumulators and post-processing constants
    dataset_num = len(labels)

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

    for idx in range(dataset_num):
        id = labels[idx]

        # Load image
        subset = "DT-0.16-dilation-1-300-epochs-free-border"
        type = "coordinates"
        npz_path = f"../stats/{subset}/{type}/{id}.npz"
        data = np.load(npz_path)
        image = data["image"]
        gt = data["label"]
        exact_gt = data["exact_gt"]

        # Load prediction
        mat_file_path = os.path.join(path, f"{id}.png.mat")
        mat = loadmat(mat_file_path)
        pred = (mat["pred"]).astype(float)

        #imgs, labels, ids, cdc, cdc_distances, exact_gt = data

        #imgs = clip(imgs)
        #labels = clip(labels)

        # Feed forward
        #image = model.prepare_data(imgs)
        #pred = model(image)[0,0,:,:].detach().cpu()
        #image = image.detach().cpu()

        # Clip regions outside of ROI
        #pred = tensor_to_image(pred)
        #image = tensor_to_image(image)
        #gt = tensor_to_image(labels)
        #exact_gt = tensor_to_image(exact_gt)

        # Post-process
        pred = mask_array(pred, image, better_mask=True, fill=1) # 0 in DT denotes cone!
        proc = hamwoods_postprocessing(pred, 0.5, verbose=True, return_coordinates=True, is_dt=False) # Note 0.5 instead of 0.0
        #proc = minimum_postprocessing(pred, verbose=True, return_coordinates=True)
        pred = mask_array(pred, image, better_mask=True, fill=0) # Undo masking with 1

        # Remove regions around image montage for evaluation
        #proc = mask_array(proc, image, better_mask=True, fill=0)
        gt = (mask_array(gt, image, better_mask=True, fill=1) == 0) # 0 in DT denotes cone!

        # Get CDC (or approximate CDC)
        if use_cdc:
            #cy, cx = cdc[0][0] # array of arrays in .npz file
            #cx, cy = cdc[0][0] # IMPORTANT: CDC is in order (x,y) + array of arrays in .npz file
            #cy, cx = cy.item(), cx.item() # cdc is Tensor
            # Load cdc
            cdc20 = np.load(f"F:/Data/ConeDatasets/20230301_cross_eroded/test/{id}.npz")["cdc20"][0]
            cx, cy = cdc20
        else:
            # Estimate CDC by mean and get distance mask
            estimated_cdc = np.mean(np.argwhere(gt == 1), axis=0)
            cy, cx = estimated_cdc

        # Get distance mask for evaluation
        dmask = get_distance_mask(image.shape, cy, cx, initial_dist) # Should not be smaller than sqrt(2) to allow diagonal offsets
        
        # Compute TPR, FDR, Dice and Chamfer for regions around CDC
        pixels_per_degree = 600 # 600 is specific to UKB dataset

        # Get TP, FP, FN masks
        tp, fp, fn, chamfer = get_tp_fp_fn_chamfer_coordinates(proc, exact_gt, image, dist_criterium_map=dmask, free_border=free_border)

        #extent = get_maximum_extent(gt.shape[0], gt.shape[1], cy, cx, pixels_per_degree) 
        extent = get_maximum_extent_from_bounding_box(image, cy, cx, pixels_per_degree)
        #extent = np.min([extent, 15]) # Cut-off
        extent = int(np.ceil(extent)) # Ceil to next integer extent#

        gt_mask = np.sqrt((exact_gt[:,0] - cy) ** 2 + (exact_gt[:,1] - cx) ** 2)
        tp_mask = np.sqrt((tp[:,0] - cy) ** 2 + (tp[:,1] - cx) ** 2)
        fp_mask = np.sqrt((fp[:,0] - cy) ** 2 + (fp[:,1] - cx) ** 2)
        fn_mask = np.sqrt((fn[:,0] - cy) ** 2 + (fn[:,1] - cx) ** 2)
        ch_mask = np.sqrt((chamfer[:,0] - cy) ** 2 + (chamfer[:,1] - cx) ** 2)

        for eccentricity in range(extent):
            low, high = arcmin_to_pixels(pixels_per_degree, eccentricity), arcmin_to_pixels(pixels_per_degree, eccentricity + 1)
            gt_ecc = np.argwhere(np.logical_and(gt_mask >= low, gt_mask <= high))
            tp_ecc = np.argwhere(np.logical_and(tp_mask >= low, tp_mask <= high))
            fp_ecc = np.argwhere(np.logical_and(fp_mask >= low, fp_mask <= high))
            fn_ecc = np.argwhere(np.logical_and(fn_mask >= low, fn_mask <= high))
            ch_ecc = np.argwhere(np.logical_and(ch_mask >= low, ch_mask <= high))

            gt_extent = exact_gt[gt_ecc]
            tp_extent = tp[tp_ecc]
            fp_extent = fp[fp_ecc]
            fn_extent = fn[fn_ecc]
            chamfer_extent = chamfer[ch_ecc,2] # Chamfer contains (y,x,chamfer_dist)!
            #print(len(gt_extent), len(tp_extent), len(fp_extent), len(fn_extent), len(chamfer_extent))

            tpr, fdr, dice, chamfer_dist = evaluate_coordinates(tp_extent, fp_extent, fn_extent, chamfer_extent)

            gt_log = len(gt_extent)
            pred_log = len(tp_extent) + len(fn_extent)
            norm = np.pi * (2.0 * eccentricity + 1)
            gt_log = gt_log / norm * 60 ** 2 # arcmin^2 to deg^2
            pred_log = pred_log / norm * 60 ** 2 #arcmin^2 to deg^2

            tprs[idx,eccentricity] = tpr
            fdrs[idx,eccentricity] = fdr
            dices[idx,eccentricity] = dice
            chamfers[idx,eccentricity] = chamfer_dist

            gt_cones[idx,eccentricity] = len(gt_extent)
            gt_density[idx,eccentricity] = gt_log
            pred_density[idx,eccentricity] = pred_log  

        # Compute local mean distances
        gt_mean_distance[idx], _ = evaluate_distances_by_eccentricity_coordinates(exact_gt, cy, cx)
        pred_mean_distance[idx], _ = evaluate_distances_by_eccentricity_coordinates(proc, cy, cx)
        # gt_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(gt, cy, cx)
        # pred_mean_distance[idx] = get_mean_neighbour_distance_over_eccentricity(proc, cy, cx)

        # Whole image
        # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium=dist, hamwoods_free_zone=free_border)
        # tpr, fdr, dice, chamfer = evaluate(proc, gt, dist_criterium_map=dmask, hamwoods_free_zone=free_border)
        tpr, fdr, dice, chamfer_dist = evaluate_coordinates(tp, fp, fn, chamfer[:,2]) # Chamfer contains (y,x,chamfer_dist)!

        wi_tprs[idx] = tpr
        wi_fdrs[idx] = fdr
        wi_dices[idx] = dice
        wi_chamfers[idx] = chamfer_dist

        # Optionally save data
        if save_data:
            # Convert arrays to save disk storage
            np.savez(f"../stats/coordinates/{id}.npz", 
                image=image.astype(float), 
                label=gt.astype(np.uint8),
                prediction=pred.astype(float),
                postprocessed=proc.astype(float), #.astype(np.uint8),
                tp=tp.astype(float), # .astype(np.uint8),
                fp=fp.astype(float), #.astype(np.uint8),
                fn=fn.astype(float), #.astype(np.uint8),
                exact_gt=exact_gt.astype(float),
                chamfer=chamfer.astype(float))

        # Clean up manually
        #del imgs, labels, ids, cdc, pred, proc, gt, exact_gt #, applied
        #torch.cuda.empty_cache()
    
    # Save stats
    np.savez("../stats/coordinates/stats_cdc.npz", tprs=tprs, fdrs=fdrs, dices=dices, chamfers=chamfers, gt_cones=gt_cones, 
        gt_density=gt_density, pred_density=pred_density, gt_mean_distance=gt_mean_distance, pred_mean_distance=pred_mean_distance)
    np.savez("../stats/coordinates/stats_whole_image.npz", wi_tprs=wi_tprs, wi_fdrs=wi_fdrs, wi_dices=wi_dices, wi_chamfers=wi_chamfers)
