import os
import matplotlib.pyplot as plt

import numpy as np

from utils.image import mark_tp_fp_fn_grid, mark_tp_fp_fn_coordinates

def save_raw_grid(path, id, image, prediction, label, postprocessed, tp, fp, fn):
    """
    Save image, label, prediction, postprocessed and deviation 
    up to true scale
    """
    # Save image
    path_image = os.path.join(path, f"{id}_image.png")
    plt.imsave(path_image, image, cmap="gray")

    # Save label
    path_label = os.path.join(path, f"{id}_label_binary.png")
    plt.imsave(path_label, label, cmap="viridis")

    # Save prediction
    path_prediction = os.path.join(path, f"{id}_prediction_DT.png")
    plt.imsave(path_prediction, prediction, cmap="viridis")

    # Save postprocessed
    path_postprocessed = os.path.join(path, f"{id}_postprocessed_binary.png")
    plt.imsave(path_postprocessed, postprocessed, cmap="viridis")

    # Save deviation between masked binary label and postprocessed prediction
    path_difference = os.path.join(path, f"{id}_deviation_binary.png")
    plt.imsave(path_difference, postprocessed - label, cmap="seismic")

    # Save marked image (TP, FP, FN)
    marked_image = mark_tp_fp_fn_grid(image, tp, fp, fn)
    path_marked = os.path.join(path, f"{id}_marked_TP_FP_FN.png")
    plt.imsave(path_marked, marked_image)

def save_raw_coordinates(path, id, image, prediction, label, postprocessed, tp, fp, fn):
    """
    Save image, label, prediction, postprocessed and deviation 
    up to true scale
    """
    def round_coordinates(coordinates):
        locs = np.round(coordinates).astype(int)
        return locs[:,0], locs[:,1]

    # Save image
    path_image = os.path.join(path, f"{id}_image.png")
    plt.imsave(path_image, image, cmap="gray")

    # Save label
    path_label = os.path.join(path, f"{id}_label_binary.png")
    temp = np.zeros_like(image)
    temp[round_coordinates(label)] = 1
    plt.imsave(path_label, temp, cmap="viridis")

    # Save prediction
    path_prediction = os.path.join(path, f"{id}_prediction_DT.png")
    plt.imsave(path_prediction, prediction, cmap="viridis")

    # Save postprocessed
    path_postprocessed = os.path.join(path, f"{id}_postprocessed_binary.png")
    temp = np.zeros_like(image)
    temp[round_coordinates(postprocessed)] = 1
    plt.imsave(path_postprocessed, temp, cmap="viridis")

    # Save deviation between masked binary label and postprocessed prediction
    #path_difference = os.path.join(path, f"{id}_deviation_binary.png")
    #plt.imsave(path_difference, postprocessed - label, cmap="seismic")

    # Save marked image (TP, FP, FN)
    marked_image = mark_tp_fp_fn_coordinates(image, tp, fp, fn)
    path_marked = os.path.join(path, f"{id}_marked_TP_FP_FN.png")
    plt.imsave(path_marked, marked_image)


@DeprecationWarning
def save(path, id, image, prediction, label, postprocessed, format="DT"):
    """
    Save image, (masked) label, (masked) prediction with
    color bar as well as axes and legend
    """
    # # Save image
    # path_image = os.path.join(path, f"{id}_image.png")
    # fig = plt.figure(dpi=300)
    # plt.pcolormesh(image, cmap="gray")
    # plt.title(f"Image {id}")
    # plt.colorbar(label="Intensity")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.tight_layout()
    # fig.savefig(path_image)
    # plt.close()

    # # Save label
    # path_label = os.path.join(path, f"{id}_label.png")
    # fig = plt.figure(dpi=300)
    # plt.pcolormesh(label, cmap="viridis")
    # plt.title(f"Label {id}")
    # plt.colorbar(label="$P_{cone}$")
    # plt.clim(0, 1)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.tight_layout()
    # fig.savefig(path_label)
    # plt.close()

    # # Save prediction
    # path_prediction = os.path.join(path, f"{id}_prediction.png")
    # fig = plt.figure(dpi=300)
    # plt.pcolormesh(prediction, cmap="viridis")
    # plt.title(f"Prediction {id}")
    # plt.colorbar(label="DT")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.tight_layout()
    # fig.savefig(path_prediction)
    # plt.close()

    # # Use common color bar for masked label and prediction
    # label = mask_array(label, image, True)
    # prediction = mask_array(prediction, image, True)

    # # Save postprocessed
    # path_postprocessed = os.path.join(path, f"{id}_postprocessed.png")
    # fig = plt.figure(dpi=300)
    # plt.pcolormesh(postprocessed, cmap="viridis")
    # plt.imshow(postprocessed, cmap="viridis")
    # plt.title(f"Postprocessed Prediction {id}")
    # plt.colorbar(label="$P_{cone}$")
    # plt.clim(0, 1)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.tight_layout()
    # fig.savefig(path_postprocessed)
    # plt.close()

    # # Save difference between masked binary label and postprocessed
    # path_difference = os.path.join(path, f"{id}_difference.png")
    # fig = plt.figure(dpi=300)
    # difference = postprocessed - label
    # plt.pcolormesh(difference, cmap="seismic")
    # plt.title(f"Difference of Post-proc. Prediction and Label {id}")
    # plt.colorbar(label="$\Delta P_{cone}$")
    # plt.clim(-1, 1)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.tight_layout()
    # fig.savefig(path_difference)
    # plt.close()
    pass