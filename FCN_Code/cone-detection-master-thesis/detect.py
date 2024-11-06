import numpy as np
import csv

from components.fcn import FCN
from components.postprocessing import coordinates_to_mask, minimum_postprocessing

from utils.particles import ParticleBasedPostProcessing
from utils.image import clip_image, read_image, erode_image, tensor_to_image, image_to_tensor, mask_array 

import matplotlib.pyplot as plt
import argparse
import pathlib
import torch
import traceback
import sys

pretrained_model_path = "pretrained_models/DT-0.16-dilation-1-300-epochs.pth"

def detect(image : np.ndarray, mode : str = "DT", postprocessing : str = "PBPP"):
    """
    Detect cones in an retinal image

    image: retinal image in which cones shall be detected (shape should be divisible by 8 to use pretrained models)

    mode: currently only 'DT'

    postprocessing: current only 'PBPP'

    Returns a 2D array of cone locations
    """

    assert image.shape[0] % 8 == 0 and image.shape[1] % 8 == 0

    fcn = None

    if mode.upper() == "DT":
        fcn = FCN(None, lr=3e-4, depth=3, input_channels=1, initial_feature_maps=32, blocks_per_resolution_layer=2, use_MSE_loss=True, masked_MSE_loss=True)
        fcn.load(pretrained_model_path)
    else:
        raise "Unknown mode"

    # Switch to evaluation mode
    fcn.eval()
    with torch.no_grad():
        # Feed forward
        image = image_to_tensor(image).to(fcn.device)
        pred = fcn(image)[0,0,:,:].detach().cpu()
        image = image.detach().cpu()

    # Clip regions outside of ROI
    pred = tensor_to_image(pred)
    image = tensor_to_image(image)

    if postprocessing.upper() == "PBPP":
        # Extract local minima
        pred = mask_array(pred, image, better_mask=True, fill=1) # 0 in DT denotes cone!
        proc = minimum_postprocessing(pred, verbose=True, return_coordinates=True)
        pred = mask_array(pred, image, better_mask=True, fill=0) # Undo masking with 1

        # Particle-based post-processing
        step_size = 0.1 #0.1 #0.2 #0.05
        alpha = 0.6 #0.75 #0.6 #0.8 #0.4
        beta = 1.0
        gamma = 1.0
        loops = 3 #3 #4 #5 #3
        steps = 8 #8 #10
        pbpp = ParticleBasedPostProcessing(proc, pred, image, None, alpha, beta, gamma, step_size, interpolation_mode="fast_quintic_b_splines")
        pbpp.postprocess(loops=loops, steps=steps, progress_bar=True, verbose=False, add_particles=True, remove_particles=True) #progress_bar=False, verbose=True)

        cones = pbpp.particles.positions
    else:
        raise "Unknown postprocessing"

    return cones, pred

def create_parser():
    """
    Creates parser of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help = "Input image")
    parser.add_argument("-o", "--output", type=str, help = "Output path")
    parser.add_argument("-matlab", help = "Save coordinates in matlab format", required = False, action='store_true')

    return parser

if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Get image from .npz file
        # image_path = "data/BAK1012L.npz"
        # image = np.load(image_path)["image"] / 255.0

        # open image
        # image values MUST BE in [0, 1] limits, otherwise CNN will produce garbadge!
        # image = read_image("image1.tif") / 255.0
        image = read_image(args.input) / 255.0

        print("Clipping...")
        image = clip_image(image, clipping_factor=8)

        print("Eroding...")
        image, applied = erode_image(image, iterations=2)

        print("Detecting and Post-Processing...")
        cones, raw = detect(image, mode="DT", postprocessing="PBPP")

    except Exception as e:
        traceback.print_exc()
        output_path_image = ''
        output_path = ''
        result = False
        sys.exit(1)
    else:
        result = True
        # save image
        output_path_image = args.output + '/' + pathlib.Path(args.input).stem + '_probMap.png'
        plt.imsave(output_path_image, raw, cmap='Greys')

        # save coords
        coordList = cones
        if args.matlab:
            for i in range(len(coordList)):
                coordList[i] = [coordList[i][1] + 1, coordList[i][0] + 1]

        output_path = args.output + '/' + pathlib.Path(args.input).stem + '.csv'
        with open(output_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)

            for data in coordList:
                # write a row to the csv file
                writer.writerow(data)
    # cones = mask_array(cones, image, better_mask=True, fill=0)

    # print("Showing detected cones...")
    # plt.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    # plt.scatter(cones[:,1], cones[:,0], c="purple", s=1, marker=".")
    # plt.show()
