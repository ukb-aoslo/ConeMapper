import numpy as np
import csv

from components.fcn import FCN
from components.postprocessing import naive_postprocessing, hamwoods_postprocessing, particle_postprocessing

from utils.image import clip_image, read_image, mark_cones, show_image, erode_image, tensor_to_image, image_to_tensor, mask_array

import matplotlib.pyplot as plt
import argparse
import pathlib
import os


def detect(image : np.ndarray, mode : str, postprocessing : str):
    """
    Detect cones in an retinal image

    image: retinal image in which cones shall be detected (shape should be divisible by 8 to use pretrained models)

    mode: currently only 'DT'

    postprocessing: choose 'naive', or 'hamwood'

    Returns a 2D array of cone locations
    """

    assert image.shape[0] % 8 == 0 and image.shape[1] % 8 == 0

    fcn = None

    if mode.upper() == "DT":
        fcn = FCN(None, lr=3e-4, depth=3, input_channels=1, initial_feature_maps=32, blocks_per_resolution_layer=2, use_MSE_loss=True, masked_MSE_loss=True)
        fcn.load("pretrained_models/5-fold-eroded-excluded_BAK8095L-DT-0.24.pth")
        threshold = 1.5 # Should be reasonable given the model's performance
    else:
        raise "Unknown mode"

    # Feed forward
    image = fcn.prepare_data(image).unsqueeze(0).unsqueeze(0)
    pred = fcn(image)[0,0,:,:].detach().cpu()
    image = image.detach().cpu()

    # Clip regions outside of ROI
    pred = tensor_to_image(pred)
    image = tensor_to_image(image)

    # Post-processing
    if postprocessing.upper() == "NAIVE":
        pred = mask_array(pred, image, better_mask=True, fill=threshold + 1)
        # TODO add pos return
        mask = naive_postprocessing(pred, threshold)

    if postprocessing.upper() == "HAMWOOD":
        pred = mask_array(pred, image, better_mask=True, fill=1)
        mask, pos = hamwoods_postprocessing(pred, threshold, verbose=True)

    return mask, pred, pos

def create_parser():
    """
    Creates parser of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help = "Input image")
    parser.add_argument("-t", "--type", help = "Type of NN. Possible types: WBCE, DT, DAFL")
    parser.add_argument("-o", "--output", type=str, help = "Output path")
    parser.add_argument("-matlab", help = "Save coordinates in matlab format", required = False, action='store_true')

    return parser

if __name__ == "__main__":
    # image_path = "data/test_image_big.png"
    # image = read_image(image_path)

    # image_path = "data/BAK8044L.npz"
    # after_load = np.load(image_path)
    # image = after_load["image"] / 255.0
    # print(after_load["label"])
    #print(after_load["cdc20"])
    # show_image(after_load["background_probability"])
    # show_image(after_load["cone_probability"])

    # parse args
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # open image
        image = read_image(args.input)

        print("Clipping...")
        image = clip_image(image)

        print("Eroding...")
        image, _ = erode_image(image)

        print("Detecting and Post-Processing...")
        cones, raw, cone_pos = detect(image, mode="DT", postprocessing="HAMWOOD") 

    except Exception as e:
        print(e)
        output_path_image = ''
        output_path = ''
        result = False
    else:
        result = True
        # save image
        output_path_image = args.output + '/' + pathlib.Path(args.input).stem + '_probMap.png'
        plt.imsave(output_path_image, raw, cmap='Greys')

        # save coords
        coordList = cone_pos
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

    # print("Marking result...")
    # result = mark_cones(image, cones)

    # show_image(raw)
    # show_image(result)