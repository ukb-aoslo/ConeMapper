# Cone finder

## Description
Cone finder is the programm, based on MarkCones_CNN.
Written on Matlab R2016b App Designer.

## Functions

### Main Functions

#### Start Cone Finder Button

The main button.
Allows user to find cone locations by FastPickFind algorithm or CNN on new image, or open image with already finded set on cones.

If you will use CNN to find cone locations, be ready to wait for about 10 mins. For the first run CNN is preparing the data for about 6-8 minutes. After preparing is finished, starts the cone locating process. It takes about 1-2 minutes. For the second and all next searches CNN will not be preparing the data, because it is already stored in memory.

#### Voronoi Button

Prints Voronoi diagram above the image. When diaragram is printed, the interface of Cone Finder is disabled. Click again on button 'Voronoi' to remove the diagram and unlock the interface.

#### Save Curent Locations Button

Saves all current cone locations to .mat file.

#### Exit Button

Exit the programm. Before exit asks user to save the data.

### Image Properties

#### Next Mosaic Image Button

If several images is opened, shows next image.

#### Log Image Switch

Uses log filter on image in 'On' state.

#### Show Image Switch

Show the image in 'On' state.

#### Brightness Slider

Changes the brightness of the image.

#### Reset Brightness Button

Resets the brightness of the image to 1.0.

#### Show Grid Switch

Shows the grid on the image. The default step is +-115 pixels.

#### Show Grid Numbers Switch

Shows the number of each cell in the grid. Works only if Show Grid Switch is 'On'.

### Marks Properties

#### Cone Marks Types Radio Buttons

The types of marks which will be placed on each cone location.
Avaliable types is Dot, Circle and Cross.

#### Show Marks Options Radio Buttons

The types of cone locations that should be shown.
Auto + User - show both types.
Userdetect - show only user detected cone locations.
Userdetect - show only auto detected cone locations.

#### Highlight Autodetected Cones Switch

Highlight autodetected cones by magenta color.

#### Show Marked Locations Switch

Shows marks.

### Edit Cone Locations

#### Edit Mode Switch

Enables edit mode. In edit mode the figure interface is not availiable. Part of the interface of Cone Finder is disabled.
In edit mode user can place new marks and remove existing marks.

Hotkeys
Mouse Left Click  - place the mark
Mouse Right Click - remove the mark
'+'               - zoom in
'-'               - zoom out
↓ ↑  ← →          - pan image
Esc               - turn off Edit Mode

Warning! Hotkeys is working only in edit mode. Unfortunately, MatLab R2016b App Designer does not have necessary callbacks to make hotkeys.

#### Change Box Size Button

Changes the box on image.

#### Delete Selected Cones Button

Selects cone location and Deletes selected cone locations on image.

#### Add Conelocs Outside Box Button

??

#### Add Conelocs In Box

??