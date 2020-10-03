classdef ConeFinderApp_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        ConeFinderUIFigure             matlab.ui.Figure
        StartConeFinderButton          matlab.ui.control.StateButton
        VoronoiVNButton                matlab.ui.control.Button
        SaveCurrentLocationsSPACEButton  matlab.ui.control.Button
        ExitButton                     matlab.ui.control.Button
        ImagePropertiesPanel           matlab.ui.container.Panel
        NextMosaicImageButton          matlab.ui.control.Button
        LogImageLSwitchLabel           matlab.ui.control.Label
        LogImageLSwitch                matlab.ui.control.Switch
        ShowimageISwitchLabel          matlab.ui.control.Label
        ShowimageISwitch               matlab.ui.control.Switch
        BrightnessPanel                matlab.ui.container.Panel
        BrightnessDBSliderLabel        matlab.ui.control.Label
        BrightnessDBSlider             matlab.ui.control.Slider
        ResetBrightnessRButton         matlab.ui.control.StateButton
        GridPanel                      matlab.ui.container.Panel
        ShowGridQSwitchLabel           matlab.ui.control.Label
        ShowGridQSwitch                matlab.ui.control.Switch
        ShowGirdNumbersESwitchLabel    matlab.ui.control.Label
        ShowGirdNumbersESwitch         matlab.ui.control.Switch
        MarksPropertiesPanel           matlab.ui.container.Panel
        ConeMarksTypeButtonGroup       matlab.ui.container.ButtonGroup
        DotZButton                     matlab.ui.control.RadioButton
        CircleCButton                  matlab.ui.control.RadioButton
        CrossXButton                   matlab.ui.control.RadioButton
        ShowMarksOptionsButtonGroup    matlab.ui.container.ButtonGroup
        AutoUser3Button                matlab.ui.control.RadioButton
        Userdetect2Button              matlab.ui.control.RadioButton
        Autodetect1Button              matlab.ui.control.RadioButton
        HighlightAutodetectedCones0SwitchLabel  matlab.ui.control.Label
        HighlightAutodetectedCones0Switch  matlab.ui.control.Switch
        ShowMarkedLocationsTSwitchLabel  matlab.ui.control.Label
        ShowMarkedLocationsTSwitch     matlab.ui.control.Switch
        EditConeLocationsPanel         matlab.ui.container.Panel
        EditModeSwitchLabel            matlab.ui.control.Label
        EditModeSwitch                 matlab.ui.control.Switch
        ChangeBoxSizeButton            matlab.ui.control.Button
        DeletedSelectedConesDelButton  matlab.ui.control.Button
        AddConelocsOutsideBoxButton    matlab.ui.control.Button
        AddConelocsInBoxButton         matlab.ui.control.Button
        HelpTextAreaLabel              matlab.ui.control.Label
        HelpTextArea                   matlab.ui.control.TextArea
    end

    
    properties (Access = private)
        
    end
    
    properties (Access = public)
        % Files
        Fname;
        Pname;
        
        % Image
        I;
        Ttl;
        A1;
        ChangeMosaic;
        Mosaic = 0;
        MultipleMosaics = 0;
        
        % Voronoi
        VColorMap;
        VoronoiPlot = 1;
        
        % Cone Locations
        Conelocs = [];
        BoxPosition;
        TotalConeCount = 0;
        ConeCount = 0;
        Minlimx = 0;
        Minlimy = 0;
        Maxlimx = 0;
        Maxlimy = 0;
        
        % Mark properties
        ShowMarks = 'On';
        MarkerSize = 2;
        MarkType = '.y';
        ShowConesByDetectType = 0;
        
        % Image properties
        ShowImage = 'On';
        LogImage = 'Off';
        Brightness = 1.0;
        
        % Grid
        ShowGrid = 'Off';
        ShowGridNumbers = 'Off';
        
        % Handler of figure window
        Handler;
        IsClosing = 0;
    end
    
    methods (Access = public)
        
        % Marks cone on image
        function MarkCone(app, x, y)
            %         x = round(x); y = round(y);       % rounded until 10.09.2019
            %         x = x; y = y;                     % exact values from 11.09.2019 (also changes in next analysis steps)
            % CNN generates more exact cone
            % locations anyway
            
            if x < app.Minlimx || x > app.Maxlimx || y < app.Minlimy || y > app.Maxlimy
                app.Conelocs(app.TotalConeCount+1,1) = x; app.Conelocs(app.TotalConeCount+1,2) = y; app.Conelocs(app.TotalConeCount+1,3) = 3;
            else
                app.Conelocs(app.TotalConeCount+1,1) = x; app.Conelocs(app.TotalConeCount+1,2) = y; app.Conelocs(app.TotalConeCount+1,3) = 2;
            end
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
            app.TotalConeCount = size(app.Conelocs,1);
            
            % plot(app.Conelocs(app.Conelocs(:,3)~=0,1), app.Conelocs(app.Conelocs(:,3)~=0,2), app.MarkType, 'MarkerSize', app.MarkerSize)
            UpdateImage(app);
            % app.ConeCount = app.ConeCount+1;
            [Inr,  Num_mosaics] = size(app.ChangeMosaic);
            app.Ttl.BackgroundColor = [1 0 0];
            title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones', '  mosaic ', num2str(app.Mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
        end
        
        % Removes Mark from image
        function RemoveMark(app, x, y)
            % Clear marked cone nearest to click location
            if (size(app.Conelocs,1)>0)
                %         x = round(x); y = round(y);       % rounded until 10.09.2019 --> see button 1
                currentloc = [x y];
                %compute Euclidean distances:
                distances = sqrt(sum(bsxfun(@minus, app.Conelocs(:,1:2), currentloc).^2,2));
                %find the smallest distance and delete that index in app.Conelocs:
                app.Conelocs(distances==min(distances),:)=[];
                
                app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
                app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
                app.TotalConeCount = size(app.Conelocs,1);
                
                % After deletion, plot current set of cones
                [Inr, Num_mosaics] = size(app.ChangeMosaic);
                UpdateImage(app);
                title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones', '  mosaic ', num2str(app.Mosaic), '/', num2str(Num_mosaics)],'Interpreter','none');
            end
        end
        
        % Saves current cone locations to a .mat file
        function SaveCurrentLocations(app)
            if app.MultipleMosaics
                app.I = app.ChangeMosaic;
            end
            
            [pstr, nm, ext] = fileparts([app.Pname app.Fname]);
            
            if strcmp(ext,'.tiff')==1 || strcmp(ext,'.tif')==1
                name1 = [pstr filesep nm '_annotated.png'];
                name2 = [pstr filesep nm '_annotated.mat'];
            elseif strcmp(ext,'.mat')==1
                name1 = [pstr filesep nm '.png'];
                name2 = [app.Pname app.Fname];
            end
            
            if ~app.IsClosing
                app.Handler.figprint = gcf;
                print(app.Handler.figprint, name1, '-dpng');
            end
            
            % names of variables must be the same to names in 'save' function
            conecount = app.ConeCount;
            I = app.I;
            conelocs = app.Conelocs;
            boxposition = app.BoxPosition;
            multiple_mosaics = app.MultipleMosaics;
            save(name2, 'conecount', 'I', 'conelocs', 'boxposition', 'multiple_mosaics');
            
            app.Ttl.BackgroundColor = [0 1 0];
            
            if app.MultipleMosaics
                app.I = app.ChangeMosaic;
            end
        end
        
        % Zooms In image
        function ZoomIn(app)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            axis([ax(1)-width/3 ax(2)+width/3 ax(3)-height/3 ax(4)+height/3]);
            zoom(1/2);
        end
        
        % Zooms Out image
        function ZoomOut(app, x, y)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            axis([x-width/2 x+width/2 y-height/2 y+height/2]);
            zoom(2);
        end
        
        % Moves image to left
        function PanLeft(app)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            increment = round(width/20);
            axis([ax(1)-increment ax(2)-increment ax(3) ax(4)]);
        end
        
        % Moves image to right
        function PanRight(app)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            increment = round(width/20);
            axis([ax(1)+increment ax(2)+increment ax(3) ax(4)]);
        end
        
        % Moves image to up
        function PanUp(app)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            increment = round(height/20);
            axis([ax(1) ax(2) ax(3)-increment ax(4)-increment]);
        end
        
        % Moves image to left
        function PanDown(app)
            ax = axis;
            width=ax(2)-ax(1);
            height=ax(4)-ax(3);
            increment = round(height/20);
            axis([ax(1) ax(2) ax(3)+increment ax(4)+increment]);
        end
        
        % Updates Image ------------------------------------------------------------
        function UpdateImage(app)
            PrintImage(app);
            
            % Print cone locations
            PlotConelocs(app);
            
            if strcmp(app.ShowGrid, 'On')
                % Print Grid
                PrintGrid(app);
            end
        end
        
        % Prints image
        function PrintImage(app)
            blank = zeros(size(app.I));
            
            if strcmp(app.ShowImage, 'Off')
                % Print blank image
                cla; app.A1 = imshow(blank); hold on;
            else
                % Get image with given brightness
                J = imadjust(app.I,[],[],app.Brightness);
                
                if strcmp(app.LogImage, 'On')
                    % Get log image
                    J = GetLogImage(app, J);
                end
                
                % Print image
                cla; app.A1 = imshow(J); hold on;
            end
        end
        
        % Gets log image
        function image = GetLogImage(app, Image)
            % fix/round/ceil/floor removes message 'Warning: Integer operands are required for colon operator when used as index'
            minlimy = round(app.Minlimy+20);
            maxliny = round(app.Maxlimy-20);
            minlimx = round(app.Minlimx+20);
            maxlimx = round(app.Maxlimx-20);
            I_INbox = log(double(Image(minlimy:maxliny, minlimx:maxlimx)));
            
            I_log = log(double(Image));
            I_log = I_log - min(I_INbox(:));
            I_log = I_log./max(I_log(:));
            
            image = I_log;
        end
        
        % Prints cone locations
        function PlotConelocs(app)
            rectangle('Position', app.BoxPosition,'EdgeColor',[1 0 1]);
            
            if strcmp(app.ShowMarks, 'On')
                switch app.ShowConesByDetectType
                    case 1
                        usercones = app.Conelocs(app.Conelocs(:,3)==2,1:2);
                        plot(usercones(:,1),usercones(:,2), app.MarkType, 'MarkerSize', app.MarkerSize);
                        
                    case 2
                        autocones = app.Conelocs(app.Conelocs(:,3)==1,1:2);
                        plot(autocones(:,1),autocones(:,2),'.m')
                        
                    otherwise
                        if strcmp(app.HighlightAutodetectedCones0Switch.Value, 'Off')
                            plot(app.Conelocs(app.Conelocs(:,3)~=0,1), app.Conelocs(app.Conelocs(:,3)~=0,2), app.MarkType, 'MarkerSize', app.MarkerSize);
                        else
                            usercones = app.Conelocs(app.Conelocs(:,3)==2,1:2);
                            autocones = app.Conelocs(app.Conelocs(:,3)==1,1:2);
                            plot(autocones(:,1),autocones(:,2),'.m');
                            plot(usercones(:,1),usercones(:,2), app.MarkType, 'MarkerSize', app.MarkerSize);
                        end
                end
            end
        end
        
        % Prints Grid
        function PrintGrid(app)
            % Get sizes
            [rows, columns, numberOfColorChannels] = size(app.I);
            
            rowsNumber = floor(rows / 115);
            stepX = floor(rows / rowsNumber);
            
            colsNumber = floor(rows / 115);
            stepY = floor(rows / colsNumber);
            
            % Get grid
            linesRows = [stepX:stepX:(rows - stepX); stepX:stepX:(rows - stepX)];
            linesCols = [stepY:stepY:(columns - stepY); stepY:stepY:(columns - stepY)];
            colsX = [zeros(size(linesCols,2),1)' + 1; zeros(size(linesCols,2),1)' + columns];
            colsY = [zeros(size(linesRows,2),1)' + 1; zeros(size(linesRows,2),1)' + rows];
            
            % Print verticals
            line(colsX, linesCols, 'Color', 'w');
            % Print horizontals
            line(linesRows, colsY, 'Color', 'w');
            
            if strcmp(app.ShowGridNumbers, 'On')
                startX = 20;
                x = startX : stepX : rows;
                
                startY = 20;
                y = startY : stepY : columns;
                sx = size(x, 2);
                sy = size(y, 2);
                
                y = repmat(y, 1, sx);
                x = repelem(x, sy);
                textArray = arrayfun(@(a)num2str(a), (x-startX)/stepX*sx+(y-startY)/stepY, 'uni', 0);
                
                % Print numbers in each cell
                text(x, y, textArray, 'Color', 'w');
            end
        end
        % Updates Image ------------------------------------------------------------
        
        % Updates selection box
        function UpdateBox(app, old_boxposition)
            app.Minlimx = app.BoxPosition(1);
            app.Minlimy = app.BoxPosition(2);
            app.Maxlimx = app.Minlimx + app.BoxPosition(3);
            app.Maxlimy = app.Minlimy + app.BoxPosition(4);
            
            minlimx_old = old_boxposition(1);
            minlimy_old = old_boxposition(2);
            maxlimx_old = minlimx_old + old_boxposition(3);
            maxlimy_old = minlimy_old + old_boxposition(4);
            
            app.Conelocs(app.Conelocs(:,1)>app.Minlimx & app.Conelocs(:,1)<=minlimx_old,3)= 1;
            app.Conelocs(app.Conelocs(:,1)<app.Maxlimx & app.Conelocs(:,1)>=maxlimx_old,3)= 1;
            app.Conelocs(app.Conelocs(:,2)>app.Minlimy & app.Conelocs(:,2)<=minlimy_old,3)= 1;
            app.Conelocs(app.Conelocs(:,2)<app.Maxlimy & app.Conelocs(:,2)>=maxlimy_old,3)= 1;
            
            app.Conelocs(app.Conelocs(:,1)<=app.Minlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,1)>=app.Maxlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,2)<=app.Minlimy,3)= 0;
            app.Conelocs(app.Conelocs(:,2)>=app.Maxlimy,3)= 0;
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
            app.TotalConeCount = size(app.Conelocs,1);
            
            UpdateImage(app);
            
            ax = gca; app.Ttl = ax.Title; app.Ttl.FontSize = 14;
            app.Ttl.BackgroundColor = [1 0 0];
            [Inr,  Num_mosaics] = size(app.ChangeMosaic);
            title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones', '  mosaic ', num2str(app.Mosaic), '/', num2str(Num_mosaics)],'Interpreter','none');
            
        end
        
        % Updates User Interface
        function UpdateUI(app)
            isDisabled = 'off';
            if app.VoronoiPlot
                isDisabled = 'on';
            end
            
            editPanelDisabled = isDisabled;
            mainButtonsDisabled = 'on';
            if strcmp(app.EditModeSwitch.Value,'On')
                editPanelDisabled = 'off';
                mainButtonsDisabled = 'off';
            end
            
            set(findall(app.EditConeLocationsPanel, '-property', 'Enable'), 'Enable', editPanelDisabled)
            set(findall(app.MarksPropertiesPanel, '-property', 'Enable'), 'Enable', isDisabled)
            set(findall(app.GridPanel, '-property', 'Enable'), 'Enable', isDisabled)
            set(findall(app.ImagePropertiesPanel, '-property', 'Enable'), 'Enable', isDisabled)
            
            app.StartConeFinderButton.Enable = mainButtonsDisabled;
            app.VoronoiVNButton.Enable = mainButtonsDisabled;
            app.ExitButton.Enable = mainButtonsDisabled;
            app.SaveCurrentLocationsSPACEButton.Enable = mainButtonsDisabled;
        end
        
        % ----------------------- Voronoi Diagram -------------------------------------
        % Get color for poygon by Number of Neighbours
        function color = GetColorNN(app, logicC, numNeighborsJ, minNeighbors, maxNeighbors, neighcolors)
            if logicC
                if numNeighborsJ <= minNeighbors
                    color = neighcolors(1,:);
                elseif numNeighborsJ > maxNeighbors
                    color = neighcolors(end,:);
                elseif numNeighborsJ > minNeighbors
                    % color = neighcolors(numNeighborsJ,:);
                    color = neighcolors(numNeighborsJ - minNeighbors,:);
                end
            else
                color = [];
            end
        end
        
        % Get color for poygon by Cone Area
        function color = GetColorCA(app, logicC, coneAreaJ, max_area_pixel, idxMinMax, neighcolors)
            if logicC
                if coneAreaJ <= max_area_pixel
                    [val,pos] = min(abs(idxMinMax - coneAreaJ));
                    color = neighcolors(pos,:);
                else
                    color = neighcolors(end,:);
                end
            else
                color = [];
            end
        end
        
        % Print Voronoi diagram
        function PrintVoronoiByType(app, voronoiType)
            numberOfClosedPolygons = 0;
            if size(app.Conelocs,1)>3
                %%%%%%%%%%%%%% PREPARE THE DATA %%%%%%%%%%%%%%
                vorocones = app.Conelocs(app.Conelocs(:,3)~=0,1:2);
                vorocones = unique(vorocones, 'rows', 'stable');
                
                dt = delaunayTriangulation(vorocones);
                
                if isempty(dt)
                    return;
                end
                
                % V - verticies in polygon, C - the polygon around V
                [V,C] = voronoiDiagram(dt);
                % vorocones(:,1:2) - equal to vorocones, because vorocones n*2, where columns is x and y coordinate, rows is verticies
                
                if strcmp(voronoiType, 'Number of Neighbors')
                    min_neighbors = 4;
                    max_neighbors = 9;
                    numcolorsNN = max_neighbors-min_neighbors;  % ********* change from 15
                    
                    % corr_factorNN = numcolorsNN/max_neighbours; % to represent whole colormap over values
                    
                    idxNN = round(linspace(1, 256, numcolorsNN));
                    neighcolorsNN = app.VColorMap(idxNN,:);
                    neighcolorsNN = flipud(neighcolorsNN);
                    
                    NumNeighbors = arrayfun(@(a) size(C{a},2), 1 : length (C));
                    
                elseif strcmp(voronoiType, 'Cone Area') % VARIANTE 2: Falschfarben sind Patchgrÿÿe
                    numcolors = 25;  % ********* change from 15
                    max_area_pixel = 80;        % basis for dividing cone Area through numcolors
                    
                    idx = round(linspace(1,256,numcolors));
                    neighcolors = app.VColorMap(idx,:);
                    neighcolors = flipud(neighcolors);
                    
                    % polyarea - ÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿ
                    ConeArea = arrayfun(@(a) polyarea(V(C{a},1),V(C{a},2)), 1 : length (C));
                    idxMinMax = round(linspace(min(ConeArea) ,max_area_pixel, numcolors));
                end
                
                %%%%%%%%%%%%%% PRINT THE DIAGRAM %%%%%%%%%%%%%%
                % Find closed polygons in C (0 - open polygon, 1 - closed polygon)
                logicC = cellfun(@(a) all(a~=1), C, 'UniformOutput', true);
                
                % Find colors for closed polygones by voronoiType
                % VARIANTE 1: Falschfarben sind Anzahl Nachbarn
                if strcmp(voronoiType, 'Number of Neighbors')
                    colors = arrayfun(@(a) app.GetColorNN(logicC(a), NumNeighbors(a), min_neighbors, max_neighbors, neighcolorsNN), 1 : length(C), 'UniformOutput', false);
                elseif strcmp(voronoiType, 'Cone Area') % VARIANTE 2: Falschfarben sind Patchgrÿÿe
                    colors = arrayfun(@(a) app.GetColorCA(logicC(a), ConeArea(a), max_area_pixel, idxMinMax, neighcolors), 1 : length(C), 'UniformOutput', false);
                end
                % Remove empty colors
                testColors = cellfun(@(a) ~isempty(a), colors, 'UniformOutput', true);
                colors = colors(testColors);
                % Make matrix from cell array of vectors
                colorsTest = vertcat(colors{:});
                
                % Remove open polygones
                C = C(logicC);
                numberOfClosedPolygons = length(C);
                % Create a matrix from list of Verice inedexes (empty values filled by NaN)
                lengthesC = cellfun('length',C);
                cellLengthesC = num2cell(lengthesC);
                maxLengthesC = max(lengthesC);
                batchC = cellfun(@(v,n) [v, nan(1, maxLengthesC-n)], C, cellLengthesC, 'UniformOutput', false);
                matrixC = vertcat(batchC{:});
                
                % Plot the diagram
                patch('Faces', matrixC,'Vertices', V, 'FaceVertexCData', colorsTest, 'FaceColor', 'flat');
                
            end
            
            % Print Basic Voronoi Diagram if there are no closed polygons
            if numberOfClosedPolygons == 0
                vorocones = app.Conelocs(app.Conelocs(:,3)~=0,1:2);
                vorocones = unique(vorocones, 'rows', 'stable');
                x = vorocones(:,1);
                y = vorocones(:,2);
                [vx, vy] = voronoi(x, y);
                plot(x,y,'r+',vx,vy,'b-');
            end
        end
        
        % ----------------------------- Mosaic ----------------------------
        % Load mosaic images
        function LoadMosaic(app, fnames, nfiles)
            ref_image = fnames{1};
            refframe = imread([app.Pname ref_image]);
            
            pad = 100;
            
            %     rect1 = [206,206,301,301];
            %     rect2 = [306,306,201,201];
            rect1 = [200,200,201,201];
            rect2 = [312,200,201,201];
            rect3 = [200,312,201,201];
            rect4 = [312,312,201,201];
            %     rect5 = [308,164,79,183];
            
            [fheight, fwidth] = size(refframe);
            % rect is [start x coordinate, y coordinate, width, height]
            refcrop1 = imcrop(refframe,rect1);
            refcrop2 = imcrop(refframe,rect2);
            refcrop3 = imcrop(refframe,rect3);
            refcrop4 = imcrop(refframe,rect4);
            %     refcrop5 = imcrop(refframe,rect5);
            
            %     figure, imshow(refframe)
            %     title('Reference Frame')
            
            % sumframe=zeros(fheight+(pad*2),fwidth+(pad*2));
            % sumframebinary=ones(fheight+(pad*2),fwidth+(pad*2));
            
            for nf = 1:nfiles
                frame = imread([app.Pname fnames{nf}]);
                % rect is [start x coordinate, y coordinate, width, height]
                crop1 = imcrop(frame,rect1);
                crop2 = imcrop(frame,rect2);
                crop3 = imcrop(frame,rect3);
                crop4 = imcrop(frame,rect4);
                %         crop5 = imcrop(frame,rect5);
                
                [shift1, dump] = dftregistration(fft2(refcrop1),fft2(crop1),10);
                [shift2, dump] = dftregistration(fft2(refcrop2),fft2(crop2),10);
                [shift3, dump] = dftregistration(fft2(refcrop3),fft2(crop3),10);
                [shift4, dump] = dftregistration(fft2(refcrop4),fft2(crop4),10);
                %         [shift5, dump] = dftregistration(fft2(refcrop5),fft2(crop5),10);
                
                
                shifts(:,:,nf) = [shift1(3:4);shift2(3:4);shift3(3:4);shift4(3:4);];
                %         shifts(:,:,nf) = [shift1(3:4);shift2(3:4);];
                meanshift(nf,:) = mean(shifts(:,:,nf));
                deviation(nf) = mean(std(shifts(:,:,nf)));
                
                f = double(zeros(fheight+(pad*2),fwidth+(pad*2)));
                f(pad+1:(fheight+pad),pad+1:(fwidth+pad))=frame;
                
                deltar = -meanshift(nf,1);
                deltac = -meanshift(nf,2);
                phase = 2;
                [nr,nc]=size(f);
                Nr = ifftshift((-fix(nr/2):ceil(nr/2)-1));
                Nc = ifftshift((-fix(nc/2):ceil(nc/2)-1));
                [Nc,Nr] = meshgrid(Nc,Nr);
                g = abs(ifft2(fft2(f).*exp(1i*2*pi*(deltar*Nr/nr+deltac*Nc/nc))).*exp(-1i*phase));
                g = g./max(g(:));
                
                app.ChangeMosaic{nf} = g(101:812,101:812);
                
                %         figure, imshow(g)
                %         title(['Frame ', num2str(nf)])
                
                clear f deltar deltac Nc Nr nc nr g
            end
            
            %     figure, imshow(app.ChangeMosaic(:,:,1));
            %     imwrite(app.ChangeMosaic(:,:,1),'frame1.tif','tif','Compression','none');
            %     figure, imshow(app.ChangeMosaic(:,:,2));
            %     imwrite(app.ChangeMosaic(:,:,2),'frame2.tif','tif','Compression','none');
            %     figure, imshow(app.ChangeMosaic(:,:,3));
            %     imwrite(app.ChangeMosaic(:,:,3),'frame3.tif','tif','Compression','none');
            
        end
        
        % ------------------------- Find Cones ------------------------------
        % Finds cone locations by FastPeakFind algorithm
        function FastPeakFinder(app)
            p = FastPeakFind(app.I);
            app.Conelocs(:,1) = p(1:2:end);
            app.Conelocs(:,2) = p(2:2:end);
            app.Conelocs(:,3) = 1;
            
            app.Conelocs(app.Conelocs(:,1)<=app.Minlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,1)>=app.Maxlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,2)<=app.Minlimy,3)= 0;
            app.Conelocs(app.Conelocs(:,2)>=app.Maxlimy,3)= 0;
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            
            plot(app.Conelocs(app.Conelocs(:,3)~=0,1), app.Conelocs(app.Conelocs(:,3)~=0,2), app.MarkType, 'MarkerSize', app.MarkerSize)
            app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
            app.TotalConeCount = size(app.Conelocs,1);
        end
        
        % Find cone locations by CNN
        function CnnFinder(app)
            % DataSet = '14111';  % dataSet specifies trained CNN (grader 1 trained network - 14(x16) training images and 1(x16) validation image)
            DataSet = 'g1+cunefare';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
            % DataSet = '42';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
            
            ImageDir = [app.Pname app.Fname];
            tic;
            startTimeCNN = clock;
            fprintf('startTimeCNN: %s \n', sprintf('%g ', startTimeCNN));
            [app.Conelocs] = RunCNNnewSet_automatic(DataSet, ImageDir, app.BoxPosition, app.I);
            endTimeCNN = clock;
            fprintf('endTimeCNN: %s \n', sprintf('%g ', endTimeCNN));
            toc;
            disp(' ');
            disp(' ');
            
            app.Conelocs = unique(app.Conelocs,'rows', 'stable');
            app.Conelocs(:,3) = 1;
            
            app.Conelocs(app.Conelocs(:,1)<=app.Minlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,1)>=app.Maxlimx,3)= 0;
            app.Conelocs(app.Conelocs(:,2)<=app.Minlimy,3)= 0;
            app.Conelocs(app.Conelocs(:,2)>=app.Maxlimy,3)= 0;
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            
            plot(app.Conelocs(app.Conelocs(:,3)~=0,1), app.Conelocs(app.Conelocs(:,3)~=0,2), app.MarkType, 'MarkerSize', app.MarkerSize)
            app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
            app.TotalConeCount = size(app.Conelocs,1);
        end
        
        % Loads Data
        function LoadData(app)
            load([app.Pname app.Fname]);
            app.ConeCount = conecount;
            app.I = I;
            app.Conelocs = conelocs;
            app.BoxPosition = boxposition;
            app.MultipleMosaics = multiple_mosaics;
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            app.ConeCount = size(app.Conelocs(app.Conelocs(:,3)~=0,1),1);
            app.TotalConeCount = size(app.Conelocs,1);
            
            app.Minlimx = app.BoxPosition(1);
            app.Minlimy = app.BoxPosition(2);
            app.Maxlimx = app.Minlimx + app.BoxPosition(3);
            app.Maxlimy = app.Minlimy + app.BoxPosition(4);
            
            if app.MultipleMosaics == 1
                app.ChangeMosaic = app.I;
                app.Mosaic = 1;
                app.I = app.ChangeMosaic{1};
            else
                app.ChangeMosaic = app.I;
                app.Mosaic = 1;
            end
            
            app.MarkerSize = 4;
            
            app.Handler.fig = figure;
            imshow(app.I), title([app.Fname,'  ',num2str(app.ConeCount),' cones'],'Interpreter','none'), hold on
            ax = gca;                    % get the current axis
            app.Ttl = ax.Title;              % get the title text object
            app.Ttl.FontSize = 14;
            app.Ttl.BackgroundColor = [1 0 0];
        end
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            set(findall(app.ConeFinderUIFigure, '-property', 'Enable'), 'Enable', 'off')
            app.StartConeFinderButton.Enable = 'on';
        end

        % Value changed function: StartConeFinderButton
        function StartConeFinderButtonValueChanged(app, event)
            format shortg
            
            % Colormap for Voronoi patches
            % map = colormap('colorcube'); close(gcf);
            % numcolors = 15;  % ********* change from 15
            % idx = round(linspace(1,15,numcolors));
            % neighcolors = map(idx,:);
            % neighcolors = flipud(neighcolors);
            
            colormaps_mpl2019;
            % map = colormap(inferno); close(gcf);
            app.VColorMap = colormap(viridis); close(gcf);
            % adjust settings for colormap in VoronoiPatch function ~ line 660
            
            % User input - Select files to analyse
            % Select tiff to analyze, or mat file in case revisiting already counted tiff
            
            start_path = ([cd filesep]);
            % start_path = (['C:\Users\Jenny\Documents\Projekte\foveal_structure_and_function\Probandendaten_STUDIE\foveal9_mosaics\MARKED_files (g1+cunefare CNN Training)' filesep]);
            % start_path = (['E:\ausgelagerte_Studiendaten\788_gradedImages' filesep]);
            
            % file selection GUI
            [fnames, app.Pname] = uigetfile('*.mat;*.tiff;*.tif;', 'Select file(s) to mark', start_path, 'MultiSelect', 'on');
            
            % checks if user cancelled
            if isequal(fnames,0) || isequal(app.Pname,0)
                % display a message in the commmand window
                disp(' ');disp(' ');disp(' ');disp(' ----> User cancelled script');disp(' ');disp(' ');
                % if cancelled exit script
                return
            end
            
            % determine number of selected filess
            % sumnorms = cellstr(fnames);
            nfiles = size(fnames(:),1);
            
            if nfiles > 10
                nfiles = 1;
            end
            
            if nfiles > 1
                app.MultipleMosaics = 1;
                app.Fname = fnames{1};
                [pstr, nm, ext] = fileparts([app.Pname app.Fname]);
            else
                app.MultipleMosaics = 0;
                app.Fname = fnames;
                [pstr, nm, ext] = fileparts([app.Pname app.Fname]);
            end
            
            %% In case of Multiselect - prepare for changing the app.Mosaic image
            if app.MultipleMosaics
                LoadMosaic(app, fnames, nfiles);
            end
            
            %% Prepare file to start/continue cone marking
            if strcmp(ext,'.tiff')==1 || strcmp(ext,'.tif')==1
                
                if app.MultipleMosaics
                    app.Mosaic = 1;
                    app.I = app.ChangeMosaic{app.Mosaic};
                elseif nfiles == 1
                    app.I = imread([app.Pname app.Fname]);
                    app.ChangeMosaic{1} = app.I;
                end
                
                app.Handler.fig = figure;
                imshow(app.I), hold on
                
                box_size = questdlg('Select box size', 'Define window size for analysis', '200x200 pixel', ...
                    '512x512  pixel', '1024x1024  pixel', '1024x1024  pixel');
                if strcmp(box_size, '200x200 pixel')
                    hi = imrect(gca,[0 0 200 200]);
                elseif strcmp(box_size, '512x512  pixel')
                    hi = imrect(gca,[0 0 512 512]);
                elseif strcmp(box_size, '1024x1024  pixel')
                    hi = imrect(gca,[0 0 1024 1024]);
                end
                
                app.BoxPosition = wait(hi);
                % app.BoxPosition = [295.8507 239.8978 200 200];    % defined, preselected positions for Box can be entered here, before starting the script
                % app.BoxPosition = [107.2877 97.5342 512 512];    % defined, preselected positions for Box can be entered here, before starting the script
                app.Minlimx = app.BoxPosition(1);
                app.Minlimy = app.BoxPosition(2);
                app.Maxlimx = app.Minlimx + app.BoxPosition(3);
                app.Maxlimy = app.Minlimy + app.BoxPosition(4);
                
                PrintImage(app);
                title(app.Fname,'Interpreter','none');
                ax = gca; app.Ttl = ax.Title; app.Ttl.FontSize = 14;
                app.Ttl.BackgroundColor = [1 0 0];
                rectangle('Position', app.BoxPosition,'EdgeColor',[1 0 1]);
                
                % QUESTDLG for use of the FastPeakFinder
                ButtonName = questdlg(' Use automatic cone finding algorithm? ', ...
                    'Startup condition', ...
                    'Fast Peak Finder', 'CNN','No','No');
                switch ButtonName
                    case 'Fast Peak Finder'
                        FastPeakFinder(app);
                        
                    case 'CNN'
                        CnnFinder(app);
                        
                    case 'No'
                        app.ConeCount = 0;
                end % switch
                
                title([app.Fname,'  ',num2str(app.ConeCount),' cones'], 'Interpreter','none')
                
                if app.MultipleMosaics == 1
                    app.ChangeMosaic{1} = app.I;
                    app.Mosaic = 1;
                    % app.I = app.I(:,:, app.Mosaic);
                else
                    app.ChangeMosaic{1} = app.I;
                    app.Mosaic = 1;
                end
                
            elseif strcmp(ext,'.mat')==1
                % load data
                LoadData(app);
                UpdateImage(app);
                title([app.Fname,'  ',num2str(app.ConeCount),' cones'],'Interpreter','none');
            end
            
            UpdateUI(app);
        end

        % Button pushed function: VoronoiVNButton
        function VoronoiVNButtonPushed(app, event)
            if app.VoronoiPlot
                % Print image
                PrintImage(app);
                % Update user interface
                app.VoronoiPlot = 0;
                UpdateUI(app);
                
                % Get voronoi type
                voronoi_type = questdlg('Select coloring for Voronoi patches', 'Coloring Voronois', 'Cone Area', ...
                    'Number of Neighbors', 'Cone Area');
                
                % Print voronoi diagram
                tic;
                PrintVoronoiByType(app, voronoi_type);
                toc;
            else
                % Print image
                UpdateImage(app);
                % Update user interface
                app.VoronoiPlot = 1;
                UpdateUI(app);
            end
        end

        % Value changed function: EditModeSwitch
        function EditModeSwitchValueChanged(app, event)
            UpdateUI(app);
            while strcmp(app.EditModeSwitch.Value,'On')
                [x,y,button] = ginput(1);
                
                switch button
                    % Mark cone center by left clicking
                    case 1
                        if app.VoronoiPlot == 1
                            MarkCone(app, x, y)
                        end
                        
                        % Clear marked cone nearest to click location
                    case 3
                        if app.VoronoiPlot == 1
                            RemoveMark(app, x, y)
                        end
                        
                        % Zoom in and out with two keys "+", "-" on numpad
                    case 45
                        ZoomIn(app)
                    case 43
                        ZoomOut(app, x, y)
                        
                        % Pan image laterally with arrow keys
                    case  28 % LEFT arrow
                        PanLeft(app)
                    case 29 % RIGHT arrow
                        PanRight(app)
                    case 30 % UP arrow
                        PanUp(app)
                    case 31 % DOWN arrow
                        PanDown(app)
                        
                        % Escape key to quit count routine
                    case 27
                        app.EditModeSwitch.Value = 'Off';
                        
                        % Darken image "D"
                    case 100
                        if app.VoronoiPlot == 1
                            if (app.BrightnessDBSlider.Value < 2.0)
                                if (app.BrightnessDBSlider.Value + 0.1 <= 2.0)
                                    app.BrightnessDBSlider.Value = app.BrightnessDBSlider.Value + 0.1;
                                else
                                    app.BrightnessDBSlider.Value = 2.0;
                                end
                            end
                            newEvent.Value = app.BrightnessDBSlider.Value;
                            BrightnessDBSliderValueChanging(app, newEvent);
                        end
                        
                        % Brighten image "B"
                    case 98
                        if app.VoronoiPlot == 1
                            if (app.BrightnessDBSlider.Value > 0.0)
                                if (app.BrightnessDBSlider.Value - 0.1 >= 0.0)
                                    app.BrightnessDBSlider.Value = app.BrightnessDBSlider.Value - 0.1;
                                else
                                    app.BrightnessDBSlider.Value = 0.0;
                                end
                            end
                            newEvent.Value = app.BrightnessDBSlider.Value;
                            BrightnessDBSliderValueChanging(app, newEvent);
                        end
                        
                        % Reset image brightness "R"
                    case 114
                        if app.VoronoiPlot == 1
                            app.BrightnessDBSlider.Value = 1.0;
                            ResetBrightnessRButtonValueChanged(app, event);
                        end
                        
                        % Log filter image "L"
                    case 108
                        if app.VoronoiPlot == 1
                            if strcmp(app.LogImageLSwitch.Value, 'Off')
                                app.LogImageLSwitch.Value = 'On';
                            else
                                app.LogImageLSwitch.Value = 'Off';
                            end
                            LogImageLSwitchValueChanged(app, event);
                        end
                        
                        % Show image "I"
                    case 105
                        if app.VoronoiPlot == 1
                            if strcmp(app.ShowimageISwitch.Value, 'Off')
                                app.ShowimageISwitch.Value = 'On';
                            else
                                app.ShowimageISwitch.Value = 'Off';
                            end
                            ShowimageISwitchValueChanged(app, event);
                        end
                        
                        % Show Marks "T"
                    case 116
                        if app.VoronoiPlot == 1
                            if strcmp(app.ShowMarkedLocationsTSwitch.Value, 'Off')
                                app.ShowMarkedLocationsTSwitch.Value = 'On';
                            else
                                app.ShowMarkedLocationsTSwitch.Value = 'Off';
                            end
                            ShowMarkedLocationsTSwitchValueChanged(app, event);
                        end
                        
                        % Circle marks "C"
                    case 99
                        if app.VoronoiPlot == 1
                            app.CircleCButton.Value = ~app.CircleCButton.Value;
                            ConeMarksTypeButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Cross marks "X"
                    case 120
                        if app.VoronoiPlot == 1
                            app.CrossXButton.Value = ~app.CrossXButton.Value;
                            ConeMarksTypeButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Dot marks "Z"
                    case 122
                        if app.VoronoiPlot == 1
                            app.DotZButton.Value = ~app.DotZButton.Value;
                            ConeMarksTypeButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Autodetected marks "1"
                    case 49
                        if app.VoronoiPlot == 1
                            app.Autodetect1Button.Value = ~app.Autodetect1Button.Value;
                            ShowMarksOptionsButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Userdetected marks "2"
                    case 50
                        if app.VoronoiPlot == 1
                            app.Userdetect2Button.Value = ~app.Userdetect2Button.Value;
                            ShowMarksOptionsButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Auto + User detected marks "3"
                    case 51
                        if app.VoronoiPlot == 1
                            app.AutoUser3Button.Value = ~app.AutoUser3Button.Value;
                            ShowMarksOptionsButtonGroupSelectionChanged(app, event);
                        end
                        
                        % Highlight autodetected marks "0"
                    case 48
                        if app.VoronoiPlot == 1
                            if strcmp(app.HighlightAutodetectedCones0Switch.Value, 'Off')
                                app.HighlightAutodetectedCones0Switch.Value = 'On';
                            else
                                app.HighlightAutodetectedCones0Switch.Value = 'Off';
                            end
                            HighlightAutodetectedCones0SwitchValueChanged(app, event);
                        end
                        
                        % Show grid "Q"
                    case 113
                        if app.VoronoiPlot == 1
                            if strcmp(app.ShowGridQSwitch.Value, 'Off')
                                app.ShowGridQSwitch.Value = 'On';
                            else
                                app.ShowGridQSwitch.Value = 'Off';
                            end
                            ShowGridQSwitchValueChanged(app, event);
                        end
                        
                        % Show grid numbers "E"
                    case 101
                        if app.VoronoiPlot == 1
                            if strcmp(app.ShowGirdNumbersESwitch.Value, 'Off')
                                app.ShowGirdNumbersESwitch.Value = 'On';
                            else
                                app.ShowGirdNumbersESwitch.Value = 'Off';
                            end
                            ShowGirdNumbersESwitchValueChanged(app, event);
                        end
                        
                        % Save current locations "SPACE"
                    case 32
                        SaveCurrentLocationsSPACEButtonPushed(app, event);
                        
                        % Delete selected locations "Del" ("entf")
                    case 127
                        if app.VoronoiPlot == 1
                            DeletedSelectedConesDelButtonPushed(app, event);
                        end
                end
                
                % Plot voronoi digram for 'Cone Area', "V"
                if button == 118
                    if app.VoronoiPlot == 1
                        PrintImage(app);
                        app.VoronoiPlot = 0;
                        UpdateUI(app);
                        PrintVoronoiByType(app, 'Cone Area');
                    else
                        app.VoronoiPlot = 1;
                        UpdateUI(app);
                        UpdateImage(app);
                    end
                end
                
                % Plot voronoi digram for 'Number of Neighbors', "N"
                if button == 110
                    if app.VoronoiPlot == 1
                        PrintImage(app);
                        app.VoronoiPlot = 0;
                        UpdateUI(app);
                        PrintVoronoiByType(app, 'Number of Neighbors');
                    else
                        app.VoronoiPlot = 1;
                        UpdateUI(app);
                        UpdateImage(app);
                    end
                end
                
            end
            UpdateUI(app);
        end

        % Value changed function: ResetBrightnessRButton
        function ResetBrightnessRButtonValueChanged(app, event)
            app.BrightnessDBSlider.Value = 1.0;
            app.Brightness = 1.0;
            UpdateImage(app);
        end

        % Value changed function: LogImageLSwitch
        function LogImageLSwitchValueChanged(app, event)
            % log_image = app.LogImageLSwitch.Value
            % default = Off
            app.LogImage = app.LogImageLSwitch.Value;
            UpdateImage(app);
        end

        % Value changed function: ShowMarkedLocationsTSwitch
        function ShowMarkedLocationsTSwitchValueChanged(app, event)
            % markson = app.ShowMarkedLocationsTSwitch.Value
            % default = On
            app.ShowMarks = app.ShowMarkedLocationsTSwitch.Value;
            UpdateImage(app);
        end

        % Value changed function: ShowimageISwitch
        function ShowimageISwitchValueChanged(app, event)
            % imageon = app.ShowimageISwitch.Value
            % default = On
            app.ShowImage = app.ShowimageISwitch.Value;
            UpdateImage(app);
        end

        % Value changing function: BrightnessDBSlider
        function BrightnessDBSliderValueChanging(app, event)
            app.Brightness = event.Value;
            UpdateImage(app);
        end

        % Selection changed function: ConeMarksTypeButtonGroup
        function ConeMarksTypeButtonGroupSelectionChanged(app, event)
            if app.CircleCButton.Value
                app.MarkType = 'oy';
                app.MarkerSize = 5;
            elseif app.CrossXButton.Value
                app.MarkType = 'xy';
                app.MarkerSize = 5;
            else
                app.MarkType = '.y';
                app.MarkerSize = 2;
            end
            
            UpdateImage(app);
        end

        % Selection changed function: ShowMarksOptionsButtonGroup
        function ShowMarksOptionsButtonGroupSelectionChanged(app, event)
            if app.Userdetect2Button.Value
                app.ShowConesByDetectType = 1;
            elseif app.Autodetect1Button.Value
                app.ShowConesByDetectType = 2;
            else
                app.ShowConesByDetectType = 0;
            end
            
            UpdateImage(app);
        end

        % Value changed function: HighlightAutodetectedCones0Switch
        function HighlightAutodetectedCones0SwitchValueChanged(app, event)
            UpdateImage(app);
        end

        % Button pushed function: SaveCurrentLocationsSPACEButton
        function SaveCurrentLocationsSPACEButtonPushed(app, event)
            SaveCurrentLocations(app);
        end

        % Button pushed function: ExitButton
        function ExitButtonPushed(app, event)
            close_dialog = questdlg('Do you want to save before quitting?',...
                'Close analysis window', 'Yes', 'No', 'Yes');
            
            if strcmp(close_dialog, 'Yes')
                SaveCurrentLocations(app);
            end
            
            close(app.ConeFinderUIFigure);
        end

        % Button pushed function: ChangeBoxSizeButton
        function ChangeBoxSizeButtonPushed(app, event)
            change_box = questdlg('Select new Box parameters', 'Change Size of Box/Rectangle', 'Keep current box', ...
                'Select new 500x500p box', 'Enter new box-parameters', 'Keep current box');
            
            if strcmp(change_box, 'Keep current box')
                % do nothing
            elseif strcmp(change_box, 'Select new 500x500p box')
                old_boxposition = app.BoxPosition;
                hi = imrect(gca,[0 0 500 500]);
                app.BoxPosition = wait(hi);
                UpdateBox(app, old_boxposition);
            elseif strcmp(change_box, 'Enter new box-parameters')
                old_boxposition = app.BoxPosition;
                app.BoxPosition = input('New box-parameters ([x y x-range y-range]: ');
                UpdateBox(app, old_boxposition);
            end
        end

        % Button pushed function: DeletedSelectedConesDelButton
        function DeletedSelectedConesDelButtonPushed(app, event)
            delBox = imrect(gca);
            box_delCones = wait(delBox);
            
            mindelx = box_delCones(1);
            mindely = box_delCones(2);
            maxdelx = mindelx + box_delCones(3);
            maxdely = mindely + box_delCones(4);
            
            app.Conelocs((app.Conelocs(:,1)>=mindelx & app.Conelocs(:,1)<=maxdelx & app.Conelocs(:,2)>=mindely & app.Conelocs(:,2)<=maxdely),:)= [];
            %         app.Conelocs(app.Conelocs(:,1)<=maxdelx,:)= [];
            %         app.Conelocs(app.Conelocs(:,2)>=mindely & app.Conelocs(:,2)<=maxdely,:)= [];
            %         app.Conelocs(app.Conelocs(:,2)<=maxdely,:)= [];
            
            UpdateImage(app);
            title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none')
        end

        % Button pushed function: AddConelocsOutsideBoxButton
        function AddConelocsOutsideBoxButtonPushed(app, event)
            add_conelocs = [];
            
            fPFBox = imrect(gca);
            box_addCones = wait(fPFBox);
            
            minaddx = box_addCones(1);
            minaddy = box_addCones(2);
            maxaddx = minaddx + box_addCones(3);
            maxaddy = minaddy + box_addCones(4);
            
            %         app.Conelocs(app.Conelocs(:,1)>=minaddx&app.Conelocs(:,1)<=maxaddx&app.Conelocs(:,2)>=minaddy&app.Conelocs(:,2)<=maxaddy,3)= 3;
            %         app.Conelocs(app.Conelocs(:,1)<=maxaddx,3)= 3;
            %         app.Conelocs(app.Conelocs(:,2)>=minaddy,3)= 3;
            %         app.Conelocs(app.Conelocs(:,2)<=maxaddy,3)= 3;
            
            p = FastPeakFind(app.I(minaddy:maxaddy, minaddx:maxaddx));
            add_conelocs(:,1) = p(1:2:end)+minaddx-1;
            add_conelocs(:,2) = p(2:2:end)+minaddy-1;
            add_conelocs(:,3) = 3;
            
            app.Conelocs = [app.Conelocs; add_conelocs];
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            
            UpdateImage(app);
            title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none');
        end

        % Button pushed function: AddConelocsInBoxButton
        function AddConelocsInBoxButtonPushed(app, event)
            fPFBox = imrect(gca);
            box_addCones = wait(fPFBox);
            
            minaddx = box_addCones(1);
            minaddy = box_addCones(2);
            maxaddx = minaddx + box_addCones(3);
            maxaddy = minaddy + box_addCones(4);
            
            app.Conelocs(app.Conelocs(:,1)>=minaddx&app.Conelocs(:,1)<=maxaddx&app.Conelocs(:,2)>=minaddy&app.Conelocs(:,2)<=maxaddy,3)= 3;
            %         app.Conelocs(app.Conelocs(:,1)<=maxaddx,3)= 3;
            %         app.Conelocs(app.Conelocs(:,2)>=minaddy,3)= 3;
            %         app.Conelocs(app.Conelocs(:,2)<=maxaddy,3)= 3;
            
            %
            %         DataSet = 'g1+cunefare';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
            %         ImageDir = [pname fname];
            %         [add_conelocs] = RunCNNnewSet_automatic(DataSet, ImageDir, boxposition, I);
            %         add_conelocs(:,3) = 3;
            
            %         p = FastPeakFind(I(minaddy:maxaddy,minaddx:maxaddx));
            %         add_conelocs(:,1) = p(1:2:end)+minaddx-1;
            %         add_conelocs(:,2) = p(2:2:end)+minaddy-1;
            %         add_conelocs(:,3) = 3;
            
            %         app.Conelocs = [app.Conelocs; add_conelocs];
            
            app.Conelocs = unique(app.Conelocs, 'rows', 'stable');
            UpdateImage(app);
            title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none');
        end

        % Button pushed function: NextMosaicImageButton
        function NextMosaicImageButtonPushed(app, event)
            if app.MultipleMosaics
                [Inr,  Num_mosaics] = size(app.ChangeMosaic);
                if app.Mosaic < Num_mosaics
                    app.Mosaic = app.Mosaic+1;
                    app.I = app.ChangeMosaic{app.Mosaic};
                    
                    UpdateImage(app);
                    title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones', '  mosaic ', num2str(app.Mosaic), '/', num2str(Num_mosaics)],'Interpreter','none');
                    
                elseif app.Mosaic == Num_mosaics
                    app.Mosaic = 1;
                    app.I = app.ChangeMosaic{app.Mosaic};
                    
                    UpdateImage(app);
                    title([app.Fname,'  ',num2str(size(app.Conelocs(app.Conelocs(:,3)~=0,1),1)),' cones', '  mosaic ', num2str(app.Mosaic), '/', num2str(Num_mosaics)],'Interpreter','none');
                end
            end
        end

        % Close request function: ConeFinderUIFigure
        function ConeFinderUIFigureCloseRequest(app, event)
            app.IsClosing = 1;
            try
                close(app.Handler.fig);
            catch
            end
            delete(app);
        end

        % Value changed function: ShowGridQSwitch
        function ShowGridQSwitchValueChanged(app, event)
            app.ShowGrid = app.ShowGridQSwitch.Value;
            UpdateImage(app);
        end

        % Value changed function: ShowGirdNumbersESwitch
        function ShowGirdNumbersESwitchValueChanged(app, event)
            app.ShowGridNumbers = app.ShowGirdNumbersESwitch.Value;
            UpdateImage(app);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create ConeFinderUIFigure and hide until all components are created
            app.ConeFinderUIFigure = uifigure('Visible', 'off');
            app.ConeFinderUIFigure.Position = [100 100 717 540];
            app.ConeFinderUIFigure.Name = 'Cone Finder';
            app.ConeFinderUIFigure.Resize = 'off';
            app.ConeFinderUIFigure.CloseRequestFcn = createCallbackFcn(app, @ConeFinderUIFigureCloseRequest, true);

            % Create StartConeFinderButton
            app.StartConeFinderButton = uibutton(app.ConeFinderUIFigure, 'state');
            app.StartConeFinderButton.ValueChangedFcn = createCallbackFcn(app, @StartConeFinderButtonValueChanged, true);
            app.StartConeFinderButton.Text = 'Start ConeFinder';
            app.StartConeFinderButton.Position = [49 496 108 22];

            % Create VoronoiVNButton
            app.VoronoiVNButton = uibutton(app.ConeFinderUIFigure, 'push');
            app.VoronoiVNButton.ButtonPushedFcn = createCallbackFcn(app, @VoronoiVNButtonPushed, true);
            app.VoronoiVNButton.Position = [183 496 136 22];
            app.VoronoiVNButton.Text = 'Voronoi [V, N]';

            % Create SaveCurrentLocationsSPACEButton
            app.SaveCurrentLocationsSPACEButton = uibutton(app.ConeFinderUIFigure, 'push');
            app.SaveCurrentLocationsSPACEButton.ButtonPushedFcn = createCallbackFcn(app, @SaveCurrentLocationsSPACEButtonPushed, true);
            app.SaveCurrentLocationsSPACEButton.VerticalAlignment = 'top';
            app.SaveCurrentLocationsSPACEButton.Position = [345 496 190 22];
            app.SaveCurrentLocationsSPACEButton.Text = 'Save Current Locations [SPACE]';

            % Create ExitButton
            app.ExitButton = uibutton(app.ConeFinderUIFigure, 'push');
            app.ExitButton.ButtonPushedFcn = createCallbackFcn(app, @ExitButtonPushed, true);
            app.ExitButton.Position = [561 496 100 22];
            app.ExitButton.Text = 'Exit';

            % Create ImagePropertiesPanel
            app.ImagePropertiesPanel = uipanel(app.ConeFinderUIFigure);
            app.ImagePropertiesPanel.TitlePosition = 'centertop';
            app.ImagePropertiesPanel.Title = 'Image Properties';
            app.ImagePropertiesPanel.Position = [14 13 178 462];

            % Create NextMosaicImageButton
            app.NextMosaicImageButton = uibutton(app.ImagePropertiesPanel, 'push');
            app.NextMosaicImageButton.ButtonPushedFcn = createCallbackFcn(app, @NextMosaicImageButtonPushed, true);
            app.NextMosaicImageButton.Position = [8.5 401 162 22];
            app.NextMosaicImageButton.Text = 'Next Mosaic Image';

            % Create LogImageLSwitchLabel
            app.LogImageLSwitchLabel = uilabel(app.ImagePropertiesPanel);
            app.LogImageLSwitchLabel.HorizontalAlignment = 'center';
            app.LogImageLSwitchLabel.VerticalAlignment = 'top';
            app.LogImageLSwitchLabel.Position = [49 336 79 15];
            app.LogImageLSwitchLabel.Text = 'Log Image [L]';

            % Create LogImageLSwitch
            app.LogImageLSwitch = uiswitch(app.ImagePropertiesPanel, 'slider');
            app.LogImageLSwitch.ValueChangedFcn = createCallbackFcn(app, @LogImageLSwitchValueChanged, true);
            app.LogImageLSwitch.Position = [66 366 45 20];

            % Create ShowimageISwitchLabel
            app.ShowimageISwitchLabel = uilabel(app.ImagePropertiesPanel);
            app.ShowimageISwitchLabel.HorizontalAlignment = 'center';
            app.ShowimageISwitchLabel.VerticalAlignment = 'top';
            app.ShowimageISwitchLabel.Position = [46 270 85 15];
            app.ShowimageISwitchLabel.Text = 'Show image [I]';

            % Create ShowimageISwitch
            app.ShowimageISwitch = uiswitch(app.ImagePropertiesPanel, 'slider');
            app.ShowimageISwitch.ValueChangedFcn = createCallbackFcn(app, @ShowimageISwitchValueChanged, true);
            app.ShowimageISwitch.Position = [66 300 45 20];
            app.ShowimageISwitch.Value = 'On';

            % Create BrightnessPanel
            app.BrightnessPanel = uipanel(app.ImagePropertiesPanel);
            app.BrightnessPanel.Title = 'Brightness';
            app.BrightnessPanel.Position = [0 144 178 117];

            % Create BrightnessDBSliderLabel
            app.BrightnessDBSliderLabel = uilabel(app.BrightnessPanel);
            app.BrightnessDBSliderLabel.HorizontalAlignment = 'right';
            app.BrightnessDBSliderLabel.VerticalAlignment = 'top';
            app.BrightnessDBSliderLabel.Position = [41 82 95 15];
            app.BrightnessDBSliderLabel.Text = 'Brightness [D, B]';

            % Create BrightnessDBSlider
            app.BrightnessDBSlider = uislider(app.BrightnessPanel);
            app.BrightnessDBSlider.Limits = [0 2];
            app.BrightnessDBSlider.MajorTicks = [0 0.5 1 1.5 2];
            app.BrightnessDBSlider.ValueChangingFcn = createCallbackFcn(app, @BrightnessDBSliderValueChanging, true);
            app.BrightnessDBSlider.Position = [13 70 150 3];
            app.BrightnessDBSlider.Value = 1;

            % Create ResetBrightnessRButton
            app.ResetBrightnessRButton = uibutton(app.BrightnessPanel, 'state');
            app.ResetBrightnessRButton.ValueChangedFcn = createCallbackFcn(app, @ResetBrightnessRButtonValueChanged, true);
            app.ResetBrightnessRButton.Text = 'Reset Brightness [R]';
            app.ResetBrightnessRButton.Position = [7 7 162 22];

            % Create GridPanel
            app.GridPanel = uipanel(app.ImagePropertiesPanel);
            app.GridPanel.Title = 'Grid';
            app.GridPanel.Position = [0 0 178 145];

            % Create ShowGridQSwitchLabel
            app.ShowGridQSwitchLabel = uilabel(app.GridPanel);
            app.ShowGridQSwitchLabel.HorizontalAlignment = 'center';
            app.ShowGridQSwitchLabel.VerticalAlignment = 'top';
            app.ShowGridQSwitchLabel.Position = [47.5 68 80 15];
            app.ShowGridQSwitchLabel.Text = 'Show Grid [Q]';

            % Create ShowGridQSwitch
            app.ShowGridQSwitch = uiswitch(app.GridPanel, 'slider');
            app.ShowGridQSwitch.ValueChangedFcn = createCallbackFcn(app, @ShowGridQSwitchValueChanged, true);
            app.ShowGridQSwitch.Position = [65 98 45 20];

            % Create ShowGirdNumbersESwitchLabel
            app.ShowGirdNumbersESwitchLabel = uilabel(app.GridPanel);
            app.ShowGirdNumbersESwitchLabel.HorizontalAlignment = 'center';
            app.ShowGirdNumbersESwitchLabel.VerticalAlignment = 'top';
            app.ShowGirdNumbersESwitchLabel.Position = [22 9 132 15];
            app.ShowGirdNumbersESwitchLabel.Text = 'Show Gird Numbers [E]';

            % Create ShowGirdNumbersESwitch
            app.ShowGirdNumbersESwitch = uiswitch(app.GridPanel, 'slider');
            app.ShowGirdNumbersESwitch.ValueChangedFcn = createCallbackFcn(app, @ShowGirdNumbersESwitchValueChanged, true);
            app.ShowGirdNumbersESwitch.Position = [65 39 45 20];

            % Create MarksPropertiesPanel
            app.MarksPropertiesPanel = uipanel(app.ConeFinderUIFigure);
            app.MarksPropertiesPanel.TitlePosition = 'centertop';
            app.MarksPropertiesPanel.Title = 'Marks Properties';
            app.MarksPropertiesPanel.Position = [206 13 296 462];

            % Create ConeMarksTypeButtonGroup
            app.ConeMarksTypeButtonGroup = uibuttongroup(app.MarksPropertiesPanel);
            app.ConeMarksTypeButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @ConeMarksTypeButtonGroupSelectionChanged, true);
            app.ConeMarksTypeButtonGroup.TitlePosition = 'centertop';
            app.ConeMarksTypeButtonGroup.Title = 'Cone Marks Type';
            app.ConeMarksTypeButtonGroup.Position = [13 317 123 106];

            % Create DotZButton
            app.DotZButton = uiradiobutton(app.ConeMarksTypeButtonGroup);
            app.DotZButton.Text = 'Dot [Z]';
            app.DotZButton.Position = [11 60 57 15];
            app.DotZButton.Value = true;

            % Create CircleCButton
            app.CircleCButton = uiradiobutton(app.ConeMarksTypeButtonGroup);
            app.CircleCButton.Text = 'Circle [C]';
            app.CircleCButton.Position = [11 38 72 15];

            % Create CrossXButton
            app.CrossXButton = uiradiobutton(app.ConeMarksTypeButtonGroup);
            app.CrossXButton.Text = 'Cross [X]';
            app.CrossXButton.Position = [11 16 71 15];

            % Create ShowMarksOptionsButtonGroup
            app.ShowMarksOptionsButtonGroup = uibuttongroup(app.MarksPropertiesPanel);
            app.ShowMarksOptionsButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @ShowMarksOptionsButtonGroupSelectionChanged, true);
            app.ShowMarksOptionsButtonGroup.Title = 'Show Marks Options';
            app.ShowMarksOptionsButtonGroup.Position = [155 317 123 106];

            % Create AutoUser3Button
            app.AutoUser3Button = uiradiobutton(app.ShowMarksOptionsButtonGroup);
            app.AutoUser3Button.Text = 'Auto + User [3]';
            app.AutoUser3Button.Position = [11 60 102 15];
            app.AutoUser3Button.Value = true;

            % Create Userdetect2Button
            app.Userdetect2Button = uiradiobutton(app.ShowMarksOptionsButtonGroup);
            app.Userdetect2Button.Text = 'Userdetect [2]';
            app.Userdetect2Button.Position = [11 38 97 15];

            % Create Autodetect1Button
            app.Autodetect1Button = uiradiobutton(app.ShowMarksOptionsButtonGroup);
            app.Autodetect1Button.Text = 'Autodetect [1]';
            app.Autodetect1Button.Position = [11 16 96 15];

            % Create HighlightAutodetectedCones0SwitchLabel
            app.HighlightAutodetectedCones0SwitchLabel = uilabel(app.MarksPropertiesPanel);
            app.HighlightAutodetectedCones0SwitchLabel.HorizontalAlignment = 'center';
            app.HighlightAutodetectedCones0SwitchLabel.VerticalAlignment = 'top';
            app.HighlightAutodetectedCones0SwitchLabel.Position = [54.5 248 184 15];
            app.HighlightAutodetectedCones0SwitchLabel.Text = 'Highlight Autodetected Cones [0]';

            % Create HighlightAutodetectedCones0Switch
            app.HighlightAutodetectedCones0Switch = uiswitch(app.MarksPropertiesPanel, 'slider');
            app.HighlightAutodetectedCones0Switch.ValueChangedFcn = createCallbackFcn(app, @HighlightAutodetectedCones0SwitchValueChanged, true);
            app.HighlightAutodetectedCones0Switch.Position = [124 278 45 20];

            % Create ShowMarkedLocationsTSwitchLabel
            app.ShowMarkedLocationsTSwitchLabel = uilabel(app.MarksPropertiesPanel);
            app.ShowMarkedLocationsTSwitchLabel.HorizontalAlignment = 'center';
            app.ShowMarkedLocationsTSwitchLabel.VerticalAlignment = 'top';
            app.ShowMarkedLocationsTSwitchLabel.Position = [70.5 172 152 15];
            app.ShowMarkedLocationsTSwitchLabel.Text = 'Show Marked Locations [T]';

            % Create ShowMarkedLocationsTSwitch
            app.ShowMarkedLocationsTSwitch = uiswitch(app.MarksPropertiesPanel, 'slider');
            app.ShowMarkedLocationsTSwitch.ValueChangedFcn = createCallbackFcn(app, @ShowMarkedLocationsTSwitchValueChanged, true);
            app.ShowMarkedLocationsTSwitch.Position = [124 202 45 20];
            app.ShowMarkedLocationsTSwitch.Value = 'On';

            % Create EditConeLocationsPanel
            app.EditConeLocationsPanel = uipanel(app.ConeFinderUIFigure);
            app.EditConeLocationsPanel.TitlePosition = 'centertop';
            app.EditConeLocationsPanel.Title = 'Edit Cone Locations';
            app.EditConeLocationsPanel.Position = [514 13 193 462];

            % Create EditModeSwitchLabel
            app.EditModeSwitchLabel = uilabel(app.EditConeLocationsPanel);
            app.EditModeSwitchLabel.HorizontalAlignment = 'center';
            app.EditModeSwitchLabel.VerticalAlignment = 'top';
            app.EditModeSwitchLabel.Position = [66.5 372 60 15];
            app.EditModeSwitchLabel.Text = 'Edit Mode';

            % Create EditModeSwitch
            app.EditModeSwitch = uiswitch(app.EditConeLocationsPanel, 'slider');
            app.EditModeSwitch.ValueChangedFcn = createCallbackFcn(app, @EditModeSwitchValueChanged, true);
            app.EditModeSwitch.Position = [74 402 45 20];

            % Create ChangeBoxSizeButton
            app.ChangeBoxSizeButton = uibutton(app.EditConeLocationsPanel, 'push');
            app.ChangeBoxSizeButton.ButtonPushedFcn = createCallbackFcn(app, @ChangeBoxSizeButtonPushed, true);
            app.ChangeBoxSizeButton.Position = [9 333 175 22];
            app.ChangeBoxSizeButton.Text = 'Change Box Size';

            % Create DeletedSelectedConesDelButton
            app.DeletedSelectedConesDelButton = uibutton(app.EditConeLocationsPanel, 'push');
            app.DeletedSelectedConesDelButton.ButtonPushedFcn = createCallbackFcn(app, @DeletedSelectedConesDelButtonPushed, true);
            app.DeletedSelectedConesDelButton.Position = [8.5 299 176 22];
            app.DeletedSelectedConesDelButton.Text = 'Deleted Selected Cones [Del]';

            % Create AddConelocsOutsideBoxButton
            app.AddConelocsOutsideBoxButton = uibutton(app.EditConeLocationsPanel, 'push');
            app.AddConelocsOutsideBoxButton.ButtonPushedFcn = createCallbackFcn(app, @AddConelocsOutsideBoxButtonPushed, true);
            app.AddConelocsOutsideBoxButton.Position = [9 262 176 22];
            app.AddConelocsOutsideBoxButton.Text = 'Add Conelocs Outside Box';

            % Create AddConelocsInBoxButton
            app.AddConelocsInBoxButton = uibutton(app.EditConeLocationsPanel, 'push');
            app.AddConelocsInBoxButton.ButtonPushedFcn = createCallbackFcn(app, @AddConelocsInBoxButtonPushed, true);
            app.AddConelocsInBoxButton.Position = [8.5 227 176 22];
            app.AddConelocsInBoxButton.Text = 'Add Conelocs In Box';

            % Create HelpTextAreaLabel
            app.HelpTextAreaLabel = uilabel(app.EditConeLocationsPanel);
            app.HelpTextAreaLabel.HorizontalAlignment = 'right';
            app.HelpTextAreaLabel.VerticalAlignment = 'top';
            app.HelpTextAreaLabel.FontSize = 14;
            app.HelpTextAreaLabel.Position = [78 190 34 18];
            app.HelpTextAreaLabel.Text = 'Help';

            % Create HelpTextArea
            app.HelpTextArea = uitextarea(app.EditConeLocationsPanel);
            app.HelpTextArea.Editable = 'off';
            app.HelpTextArea.FontSize = 14;
            app.HelpTextArea.Position = [0 0 193 188];
            app.HelpTextArea.Value = {'HOTKEYS WORKS ONLY IN EDIT MODE!!!!!'; ''; 'Mouse Left Click - place Mark'; 'Mouse Right Click - remove Mark'; '''+'' - zoom in'; '''-'' - zoom out'; 'ÿ ÿ  ÿ ÿ - pan image'; 'Esc - turn off Edit Mode'};

            % Show the figure after all components are created
            app.ConeFinderUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = ConeFinderApp_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.ConeFinderUIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.ConeFinderUIFigure)
        end
    end
end