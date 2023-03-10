import os
import matplotlib.pyplot as plt
import numpy as np

from utils.image import mask_array

def save_raw(path, id, image, prediction, label):
    """
    Save image, (masked) label, (masked) prediction without
    color bar and up to true scale
    """
    # Save image
    path_image = os.path.join(path, f"{id}_image.png")
    plt.imsave(path_image, image, cmap="gray")

    # Save label
    path_label = os.path.join(path, f"{id}_label.png")
    plt.imsave(path_label, label, cmap="viridis")

    # Save masked label
    path_masked_label = os.path.join(path, f"{id}_masked_label.png")
    plt.imsave(path_masked_label, mask_array(label, image, True), cmap="viridis")

    # Save prediction
    path_prediction = os.path.join(path, f"{id}_prediction.png")
    plt.imsave(path_prediction, prediction, cmap="viridis")

    # Save masked prediction
    path_masked_prediction = os.path.join(path, f"{id}_masked_prediction.png")
    plt.imsave(path_masked_prediction, mask_array(prediction, image, True), cmap="viridis")

    # Save difference between masked label and prediction
    path_difference = os.path.join(path, f"{id}_difference.png")
    difference = prediction - label
    plt.imsave(path_difference, mask_array(difference, image, True), cmap="seismic")

def save(path, id, image, prediction, label, format="DT"):
    """
    Save image, (masked) label, (masked) prediction with
    color bar as well as axes and legend
    """
    # Save image
    path_image = os.path.join(path, f"{id}_image.png")
    fig = plt.figure(dpi=300)
    plt.imshow(image, cmap="gray")
    plt.title(f"Image {id}")
    plt.colorbar(label="Intensity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_image)
    plt.close()

    # Save label
    path_label = os.path.join(path, f"{id}_label.png")
    fig = plt.figure(dpi=300)
    plt.imshow(label, cmap="viridis")
    plt.title(f"Label {id}")
    plt.colorbar(label=format)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_label)
    plt.close()

    # Save prediction
    path_prediction = os.path.join(path, f"{id}_prediction.png")
    fig = plt.figure(dpi=300)
    plt.imshow(prediction, cmap="viridis")
    plt.title(f"Prediction {id}")
    plt.colorbar(label=format)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_prediction)
    plt.close()

    # Use common color bar for masked label and prediction
    label = mask_array(label, image, True)
    prediction = mask_array(prediction, image, True)
    min_color = np.min(np.dstack([label, prediction]))
    max_color = np.max(np.dstack([label, prediction]))

    # Save masked label
    path_masked_label = os.path.join(path, f"{id}_masked_label.png")
    fig = plt.figure(dpi=300)
    plt.imshow(label, cmap="viridis")
    plt.title(f"Masked Label {id}")
    plt.colorbar(label=format)
    plt.clim(min_color, max_color)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_masked_label)
    plt.close()

    # Save masked prediction
    path_masked_prediction = os.path.join(path, f"{id}_masked_prediction.png")
    fig = plt.figure(dpi=300)
    plt.imshow(prediction, cmap="viridis")
    plt.title(f"Masked Prediction {id}")
    plt.colorbar(label=format)
    plt.clim(min_color, max_color)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_masked_prediction)
    plt.close()

    # Save difference between masked label and prediction
    path_difference = os.path.join(path, f"{id}_difference.png")
    fig = plt.figure(dpi=300)
    difference = prediction - label
    plt.imshow(difference, cmap="seismic")
    plt.title(f"Difference of Masked Prediction and Masked Label {id}")
    plt.colorbar(label=f"Difference in {format}")
    plt.clim(-8,8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    fig.savefig(path_difference)
    plt.close()

def visualize_particles(particles):
    y, x = particles[:,0], particles[:,1]
    mean = np.mean(particles, axis=0)
    plt.figure(figsize=(3,3), dpi=300)
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    print(f"Particles - mean ({mean[0]},{mean[1]})")
    plt.show()