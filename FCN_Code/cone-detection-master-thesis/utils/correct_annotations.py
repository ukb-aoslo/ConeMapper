import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def correct_annotations(id, image, tp, fp, fn, to_correct, cy, cx, radius=32, save_images=False):  
    if to_correct == "FP":
        locations = np.argwhere(fp == 1)
    elif to_correct == "FN":
        locations = np.argwhere(fn == 1)
    elif to_correct == "TP":
        locations = np.argwhere(tp == 1)

    is_cone = np.ones(len(locations))
    eccentricities = np.zeros(len(locations))

    for idx, (y,x) in enumerate(locations):
        fig, ax = plt.subplots(num=f"{to_correct} - Is this a CONE?", dpi=150)

        # Compute eccentricity
        pixels_per_arcminute = 600 / 60 # 600px/degree only in UKB dataset!
        cone_eccentricity = np.sqrt((cy - y) ** 2 + (cx - x) ** 2) / pixels_per_arcminute 
        eccentricities[idx] = cone_eccentricity

        plt.title(f"Image {idx+1} / {len(locations)} - {id}\nChecking {to_correct} in the center\nEccentricity: {cone_eccentricity:0.02f} [arcmin]")

        # Show image
        fig.subplots_adjust(bottom=0.2)
        crop = image[y-radius:y+radius+1,x-radius:x+radius+1]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        # Show different marks
        if to_correct != "FP":
            fp_crop = fp[y-radius:y+radius+1,x-radius:x+radius+1]
            temp = np.argwhere(fp_crop == 1)
            for (ty, tx) in temp:
                ax.scatter(tx, ty, s=radius//2, alpha=0.7, c="red")

        if to_correct != "FN":
            fn_crop = fn[y-radius:y+radius+1,x-radius:x+radius+1]
            temp = np.argwhere(fn_crop == 1)
            for (ty, tx) in temp:
                ax.scatter(tx, ty, s=radius//2, alpha=0.7, c="blue")

        if to_correct != "TP":
            tp_crop = tp[y-radius:y+radius+1,x-radius:x+radius+1]
            temp = np.argwhere(tp_crop == 1)
            for (ty, tx) in temp:
                ax.scatter(tx, ty, s=radius//2, alpha=0.7, c="green")

        # Mark center
        if to_correct == "FP":
            center_color = "red"
        if to_correct == "FN":
            center_color = "blue"
        if to_correct == "TP":
            center_color = "green"
        ax.scatter(radius, radius, s=radius//2, alpha=0.7, c=center_color)

        # Buttons: CONE / NO CONE
        def next(event):
            plt.close()

        def toggle(event):
            is_cone[idx] = 0
            plt.close()

        axprev = fig.add_axes([0.44, 0.02, 0.1, 0.075])
        axnext = fig.add_axes([0.56, 0.02, 0.1, 0.075])

        bnext = Button(axnext, "NO CONE", color="red")
        bnext.on_clicked(toggle)
        bprev = Button(axprev, f'CONE', color="green")
        bprev.on_clicked(next)

        ax.set_xlabel("TP: Green - FP: Red - FN: Blue")
        plt.show(block=True)

        if save_images:
            color_crop = np.stack([crop, crop, crop], axis=0) # Color image
            if to_correct != "TP": # Add TP in green
                color_crop[0,:,:] = (1 - tp_crop) * color_crop[0,:,:]
                color_crop[1,:,:] += tp_crop
                color_crop[2,:,:] = (1 - tp_crop) * color_crop[2,:,:]
            else:
                color_crop[:,radius,radius] = np.array([0, 1, 0])

            if to_correct != "FP": # Add FP in red
                color_crop[0,:,:] += fp_crop
                color_crop[1,:,:] = (1 - fp_crop) * color_crop[1,:,:]
                color_crop[2,:,:] = (1 - fp_crop) * color_crop[2,:,:]
            else:
                color_crop[:,radius,radius] = np.array([1, 0, 0])
            
            if to_correct != "FN": # Add FN in blue
                color_crop[0,:,:] = (1 - fn_crop) * color_crop[0,:,:]
                color_crop[1,:,:] = (1 - fn_crop) * color_crop[1,:,:]
                color_crop[2,:,:] += fn_crop
            else:
                color_crop[:,radius,radius] = np.array([0, 0, 1])

            color_crop = np.clip(color_crop, 0, 1)
            color_crop = color_crop.transpose((1,2,0))
            plt.imsave(f"results/images/{id}_{to_correct}_{idx:03d}.png", color_crop)

    np.savez(f"results/{id}_{to_correct}.npz", is_cone=is_cone, eccentricities=eccentricities)

if __name__ == "__main__":
    # Labels of montages within the test set
    labels = ["BAK1012L", "BAK1021R", "BAK1034R", "BAK1040L", "BAK1040R", "BAK1041L", "BAK1064R", "BAK1086R", "BAK1090R", "BAK8015L"]
    path = "." #"../stats/DT-0.16-dilation-1-300-epochs-free-border"
    n = len(labels)

    # Load files
    for id_label, label in enumerate(labels):
        # Get data from npz file
        data = np.load(f"{path}/{label}.npz")

        # Get parts
        image = data["image"]
        gt = data["label"]
        fn = data["fn"]
        fp = data["fp"]
        tp = data["tp"]

        # Estimate CDC by mean of GT locations
        estimated_cdc = np.mean(np.argwhere(gt == 1), axis=0)
        cy, cx = estimated_cdc

        # Ensure that results folder exists
        os.makedirs(f"results/images", exist_ok=True)

        # Correct annotations of FP
        correct_annotations(label, image, tp, fp, fn, "FP", cy, cx, save_images=True)

        # Correct annotations of FN
        # correct_annotations(label, image, tp, fp, fn, "FN", cy, cx)

        # Correct annotations of TP
        # correct_annotations(label, image, tp, fp, fn, "TP", cy, cx)

    