### 14.12.2022
 - New features:
    - Add Density calculations (Euclidian N Cones, Nearest Neighbor mean, Yellots Rings)
    - Add Statistics window
    - Add Scalebar
    - Add Documentation
    - Add Units settings

 - Fixed:
    - Errors of the UI
    - Saving file errors

### 14.04.2022
 - New features:
    - Add colormaps for voronoi diagram
    - Add import/export for .csv format
    - Add old format reading

### 27.01.2022
 - Deleted:
    - "Bounding box" has been deleted.
    - "Compare with" function has been deleted.
    - "Mosaic" has been deleted.
    - Test algorithm of improving cones after CNN (by distance model) has been deleted.
    - Edit mode has been removed.
 - New features:
    - "Empty" view of Voronoi Diagram (polygons is not filled by color).
    - Voronoi Diagram is now works with every view feature of ConeFinder.
    - Cones can be edited during Voronoi Diagram presentation. Changes will be applyed to diagram immediately.
    - Minimap has been added.
    - Info label has been added. Now you can see live time coordinates and pixel value under cursor.
 - Updates:
    - Hotkeys now work all time (not only Edit mode).
    - Pan and Zoom tools was updated.
    - All switchers in the interface were changed to check boxes.
    - View logic was updated. Current presentation order: Image -> Log filter -> Laplacian of Gaussian -> Voronoi Diagram -> Conelocations -> Grid
    - Voronoi Diagram performance has been improved.
    - Everything in the code was adjusted to Matlab 2021a version.
    - Filename and cones info now in the window title.
 - Bug fixes:
    - Fixed bugs when you choose nothing in dialog windows.
    - Fixed bugs with grid presentation (orientation, was not working with large not sqare images).
    - Fixed bugs with overlaping during CNN recognition.
    - Fixed "Delete doubles" function. Not it is working with large numbers of cones.
    - Fixed Voronoi Diagram color map scaling.
    - Fixed remember last used path for fileopengui.
    - Fixed button labeling.

### 10.12.2021
 - Fixed bug with "Show grid".
 - Left-top of each cell in grid shows the starting patch of the CNN recognition process.
 - Fixed bug with empty strips after recognition.

### 27.07.2021
 - Fixed bug with not working Voronoi diagram in EditMode.

### 17.03.2021
 - Improve usability of "Corrected Cone Locs2".
 - Add script to calculate brightness statistics of cone locations(/Utilities/get_brightness_stats_all_files.m).
 - Add restriction by pixel brightness in correction algorithm. 
   If pixel brightness less than threshold value, than this point will not be consider as potential cone location.

### 08.03.2021
 - Improve usability of "Corrected Cone Locs2".
 - Add check box "Use Correted Cones" in Voronoi panel.
 - Add check box "Edit Corrected Cones" in Edit Cone Locations panel.
 - Readme is updated.
 - Fix small bugs.
 
### 01.03.2021
 - Changes from ConeFinderApp is transfered to ConeFinderApp_v2020b.
 
### 08.02.2021
 - Add new image filter Laplassian of Gaussian.
 - Add new correction algorithm. But it is still in develop.
 - Fix small bugs.

### 28.01.2021

 - Add Cone Finder App for Matlab R2020b
 - Fix voronoi diagrams in Edit Mode
 - Fix spelling mistakes in command line
