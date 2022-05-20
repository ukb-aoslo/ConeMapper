# Cone finder

## Description
Cone finder is the programm, based on MarkCones_CNN.
Written on Matlab R2021a App Designer.

## Functions

### Main Functions

#### File -> New recognition

The main button.
Allows user to find cone locations by FastPickFind algorithm or CNN on new image.
If you will use CNN to find cone locations, be ready to wait for about 10 mins. 
After preparing is finished, starts the cone locating process. It takes about 1-2 minutes. 
For the second and all next searches CNN will not be preparing the data, because it is already stored in memory.

#### File -> Open recognition

Allows user to open .mat file with recognized cone locations and image.

#### File -> Save recognition

Saves all current cone locations to .mat file.

#### File -> Exit

Exit the programm. Before exit asks user to save the data.



### Image Properties

#### Log Image

Uses log filter on image.

#### LapOfGaus

Uses Laplacian of Gaussian on image.
What is it, read here: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm

#### Show Image

Show the image.

#### Brightness Slider

Changes the brightness of the image.

#### Reset Brightness

Resets the brightness of the image to 1.0.

#### Show Grid

Shows the grid on the image. The default step is +-115 pixels.

#### Show Grid Numbers

Shows the number of each cell in the grid. Works only if Show Grid is 'On'.



### Marks Properties

#### Cone Marks Types Radio Buttons

The types of marks which will be placed on each cone location.
Avaliable types is Dot, Circle and Cross.

#### Show Marks Options Radio Buttons

The types of cone locations that should be shown.
 - Auto + User - show both types.
 - Userdetect - show only user detected cone locations.
 - Autodetect - show only auto detected cone locations.

#### Highlight Autodetected Cones

Highlight autodetected cones by magenta color.

#### Show Marked Locations

Shows marks.



### Voronoi

#### Voronoi Diagram

Prints Voronoi diagram above the image.

#### Voronoi Type Radio Buttons

The types of Voronoi diagram to show.
 - Empty - prints just Voronoi Diagram without anything else.
 - Cone Area - prints Voronoi Diagram only with closed polygons. Each polygon filled with color, depending on its area.
 - Number of Neighbors - prints Voronoi Diagram only with closed polygons. Each polygon filled with color, depending on number of neighbour cells.



### Improve CNN Results

#### Del Outside Image Button

Deletes cone locations outside the image (on black border around the image; consider pixels with lower value then 3).

#### Del Doubles Button

Deletes cone locations which has distance to neighbour less then 1,5 pixel.



### Edit Cone Locations

#### Delete Selected Cones Button

Selects cone location and Deletes selected cone locations on image.

#### Add Conelocs by fast peak find

Add Conelocs outside the Box by using FastPickFind.


### Hotkeys
 - Alt + Mouse Left Click  - place the mark
 - Alt + Mouse Right Click - remove the mark
 - Mouse Wheel - zoom in/out
 - Mouse Middle Button - pan image
 - Arrow keys        - pan image
 - D, B              - darken/brighten image
 - R                 - reset brightness
 - L                 - apply log filter
 - I                 - show image
 - T                 - show marks
 - C, X, Z           - circle/cross/dot cone marks
 - 1, 2, 3           - auto/user/auto+user detected cone locations
 - 0                 - highlight autodetected marks
 - Q                 - show grid
 - E                 - show grid number
 - SPACE             - save current locations
 - Del (entf)        - delete selected cone locations
 - V                 - voronoi diagram for 'Cone Area'
 - N                 - voronoi diagram for 'Number of Neighbors'

 ### Ploting/Filter applying order

Image -> Log filter -> Laplacian of Gaussian -> Voronoi Diagram -> Conelocations -> Grid


## Known Bugs

1. "When i'm in the cone finder marking window and use my normal Volume key on my keyboard  it changes cone marker colors while scrolling "up or down" the volume"