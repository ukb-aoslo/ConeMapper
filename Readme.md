# Cone Mapper

## Description
Cone Mapper is the programm, based on MarkCones_CNN.
Written on Matlab R2022b App Designer.
The program allows to annotate AOSLO images semi-automaticaly and make basic analysis.
The documentation is availiable in Cone Mapper: *Help -> Show documentation*, or _./Documentation/*.mlx_ files.

## Disclaimer
   Last editor: Aleksandr Gutnikov
   Last edit on: 21.10.2024
   Copyright (c) Aleksandr Gutnikov, Wolf M. Harmening, 2024 AOVisionLab

   Part of the app is using code of other people: 
 - Jenny Witten, for Euclidean distance N cones density. Ref: Jenny L. Reiniger, Niklas Domdei, Frank G. Holz, and Wolf M. Harmening. "Human gaze is systematically offset from the center of cone topography." Current Biology 31, no. 18 (2021): 4188-4193. doi: https://doi.org/10.1016/j.cub.2021.07.005
 - Julius Ameln, for Nearest Neighbour Distance cone density.
 - Patrick Hähn, for cone recognition FCN. Ref: Hähn, P. Regularity-Aware Detection of Cone Photoreceptors via CNNs and Particle Systems. Master’s thesis, Rheinische Friedrich-Wilhelms-Universität Bonn, Bonn, Germany (2023). Institute of Computer Science. First Examiner: Prof. Dr. Thomas Schultz, Second Examiner: Dr. Wolf Harmening, Supervisor: Dr. Shekoufeh Gorgi-Zadeh.
 - Robert F. Cooper; Geoffrey K. Aguirre; Jessica I. W. Morgan, for Yellots ring cone density. Ref: Robert F. Cooper, Geoffrey K. Aguirre, Jessica I. W. Morgan; Fully Automated Estimation of Spacing and Density for Retinal Mosaics. Trans. Vis. Sci. Tech. 2019;8(5):26. doi: https://doi.org/10.1167/tvst.8.5.26
 - D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, for Convolutional Neural Network for cone detection. Ref: D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017. Released under a GPL v2 license.
 - Chen Xinfeng, for scalebar. Ref: Chen Xinfeng (2022). chenxinfeng4/scalebar (https://github.com/chenxinfeng4/scalebar), GitHub. Retrieved November 10, 2022.
 - Adi Natan, for 2D Fast peak find algorithm. Ref: Natan (2022). Fast 2D peak finder (https://github.com/adinatan/fastpeakfind/releases/tag/1.13.0.0), GitHub. Retrieved November 14, 2022.
 - Benjamin Kraus, for Convolution in 1D or 2D ignoring NaNs and (optionally) correcting for edge effects. Ref: Benjamin Kraus (2022). nanconv (https://www.mathworks.com/matlabcentral/fileexchange/41961-nanconv), MATLAB Central File Exchange. Retrieved December 7, 2022.

 - Used icons: <a href="https://www.flaticon.com/free-icons/lock" title="lock icons">Lock icons created by Muhammad Ali - Flaticon</a>
