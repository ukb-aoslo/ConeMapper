% MARKCONES - Mark cones interactively on tif images
%
% Run MarkCones.m from command line, select TIF image to semi-manually
% mark cone locations. Start by selecting a rectangular selection where 
% cones are to be marked. If the question dialog is answered with 'yes', 
% FastPeakFind.m needs to be present to auto-select some cones.
% 
% SPACEBAR saves the current selection of cones into a mat file of the
% same name as the tif image selected. With restart of the script, that 
% mat file can be selected instead of a tif image to continue from the 
% last state. 
% 
% General usage:  
% LEFT MOUSE BUTTON marks the current crosshair location, 
% RIGHT MOUSE BUTTON deletes marker closest to current location
% 
% The following keyboard strokes can be used:
%
% SPACEBAR: save current locations to .mat file
% + : Zoom in
% - : Zoom out
% Arrow keys: pan image
% A: add conelocs: creates a box to transfer conelocs inside from column 3 == 0 to 3
% B: Brighten image
% D: Darken image
% I: toggle image visibility
% C: toggle image visibility and show circular cone markings 
% H: toggle image visibility and show hexagonal cone markings 
% M: toggle background mosaic (if multiple images were selected)
% R: Reset image brightness
% S: change the Size of the box/rectangle
% T: toggle marker visibility
% V: toggle Voronoi diagram (takes a short while to compute)
% entf: creates a box to remove all conelocs in selected area
% F: add conelocs using fast peak finder in specified box on the currently active mosaic (I)
% 1: show auto selected cones only
% 2: show user selected cones only
% 3: toggle between all cones and user/auto-cone visibility
%
% % save conelocs (column 3)
%   1 - auto selected cones inside the box (CNN - in older versions: FastPeakFinder)
%   2 - user selected cones inside the box 
%   3 - user selected cones outside the box
%   0 - auto selected outside the box 
% 
% wmharmening@gmail.com --- http://ao.ukbonn.de
% original function: MarkConesRectangle
% last changes: 2019-04-04 (by JLR) 

%% Startup parameters
clear all, close all, clc;
clear

gamma = 1.0;
log_image = 1;
markson = 1;
imageon = 1;
bothcones = 1;
voronoiplot = 1;
conelocs = [];
% Colormap for Voronoi patches
% map = colormap('colorcube'); close(gcf);
% numcolors = 15;  % ********* change from 15 
% idx = round(linspace(1,15,numcolors));
% neighcolors = map(idx,:);
% neighcolors = flipud(neighcolors);

colormaps_mpl2019
% map = colormap(inferno); close(gcf);
map = colormap(viridis); close(gcf);
% adjust settings for colormap in VoronoiPatch function ~ line 660

%% User input - Select files to analyse
% Select tiff to analyze, or mat file in case revisiting already counted tiff

start_path = ([cd filesep]);
% start_path = (['C:\Users\Jenny\Documents\Projekte\foveal_structure_and_function\Probandendaten_STUDIE\foveal9_mosaics\MARKED_files (g1+cunefare CNN Training)' filesep]);
% start_path = (['E:\ausgelagerte_Studiendaten\788_gradedImages' filesep]);

[fnames, pname] = uigetfile('*.mat;*.tiff;*.tif;', 'Select file(s) to mark', start_path, 'MultiSelect', 'on');   %file selection GUI
if isequal(fnames,0) || isequal(pname,0) % checks if user cancelled
    disp(' ');disp(' ');disp(' ');disp(' ----> User cancelled script');disp(' ');disp(' '); %display a message in the commmand window
    return % if cancelled exit script
end
sumnorms = cellstr(fnames); nfiles = size(fnames(:),1); % determine number of selected files

if nfiles > 10
    nfiles = 1
end

if nfiles > 1 
    multiple_mosaics = 1;
    fname = fnames{1};
    [pstr, nm, ext] = fileparts([pname fname]);
else
    multiple_mosaics = 0;
    fname = fnames;
    [pstr, nm, ext] = fileparts([pname fname]);
end
%% In case of Multiselect - prepare for changing the mosaic image
if multiple_mosaics

    ref_image=fnames{1};
    refframe = imread([pname ref_image]);
    
    pad = 100;

%     rect1 = [206,206,301,301];
%     rect2 = [306,306,201,201];
    rect1 = [200,200,201,201];
    rect2 = [312,200,201,201];
    rect3 = [200,312,201,201];
    rect4 = [312,312,201,201];
%     rect5 = [308,164,79,183];

    [fheight, fwidth] = size(refframe);
    refcrop1 = imcrop(refframe,rect1); % rect is [start x coordinate, y coordinate, width, height]
    refcrop2 = imcrop(refframe,rect2);
    refcrop3 = imcrop(refframe,rect3);
    refcrop4 = imcrop(refframe,rect4);
%     refcrop5 = imcrop(refframe,rect5);

%     figure, imshow(refframe)
%     title('Reference Frame')

    sumframe=zeros(fheight+(pad*2),fwidth+(pad*2));
    sumframebinary=ones(fheight+(pad*2),fwidth+(pad*2));


    for nf = 1:nfiles

        frame = imread([pname fnames{nf}]);
        crop1 = imcrop(frame,rect1); % rect is [start x coordinate, y coordinate, width, height]
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

        change_mosaic{nf} = g(101:812,101:812);

%         figure, imshow(g)
%         title(['Frame ', num2str(nf)])
        
        clear f deltar deltac Nc Nr nc nr g
    end
    
%     figure, imshow(change_mosaic(:,:,1));
%     imwrite(change_mosaic(:,:,1),'frame1.tif','tif','Compression','none');
%     figure, imshow(change_mosaic(:,:,2));
%     imwrite(change_mosaic(:,:,2),'frame2.tif','tif','Compression','none');
%     figure, imshow(change_mosaic(:,:,3));
%     imwrite(change_mosaic(:,:,3),'frame3.tif','tif','Compression','none');
    
end
%% Prepare file to start/continue cone marking 

if strcmp(ext,'.tiff')==1 || strcmp(ext,'.tif')==1
    
    if multiple_mosaics
        mosaic = 1;
        I = change_mosaic{mosaic};
    elseif nfiles == 1
        I = imread([pname fname]);
        change_mosaic{1} = I;
    end
    
    h.fig = figure; imshow(I), hold on
    box_size = questdlg('Select box size', 'Define window size for analysis', '200x200 pixel', ...
            '512x512  pixel', '1024x1024  pixel', '1024x1024  pixel');
        if strcmp(box_size, '200x200 pixel')
            hi = imrect(gca,[0 0 200 200]);
            Marker_Size = 2;
        elseif strcmp(box_size, '512x512  pixel')
            hi = imrect(gca,[0 0 512 512]);
            Marker_Size = 2;
        elseif strcmp(box_size, '1024x1024  pixel')
            hi = imrect(gca,[0 0 1024 1024]);
            Marker_Size = 2;
        end
    boxposition = wait(hi);
%     boxposition = [295.8507 239.8978 200 200];    % defined, preselected positions for Box can be entered here, before starting the script 
%     boxposition = [107.2877 97.5342 512 512];    % defined, preselected positions for Box can be entered here, before starting the script 
    minlimx = boxposition(1);
    minlimy = boxposition(2);
    maxlimx = minlimx + boxposition(3);
    maxlimy = minlimy + boxposition(4);
    
    cla; a1 = imshow(I);title(fname,'Interpreter','none'), hold on;
    ax = gca; ttl = ax.Title; ttl.FontSize = 14;
    ttl.BackgroundColor = [1 0 0];
    rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        
    % QUESTDLG for use of the FastPeakFinder
    ButtonName = questdlg(' Use automatic cone finding algorithm? ', ...
        'Startup condition', ...
        'Fast Peak Finder', 'CNN','No','No');
    switch ButtonName,
        case 'Fast Peak Finder',
            p = FastPeakFind(I);
            conelocs(:,1) = p(1:2:end);
            conelocs(:,2) = p(2:2:end);
            conelocs(:,3) = 1;
            
            conelocs(conelocs(:,1)<=minlimx,3)= 0;
            conelocs(conelocs(:,1)>=maxlimx,3)= 0;
            conelocs(conelocs(:,2)<=minlimy,3)= 0;
            conelocs(conelocs(:,2)>=maxlimy,3)= 0;
            
            conelocs = unique(conelocs, 'rows', 'stable');
            
            plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            conecount = size(conelocs(conelocs(:,3)~=0,1),1);
            totalconecount = size(conelocs,1);
            
        case 'CNN',
            p = FastPeakFind(I);

%             DataSet = '14111';  % dataSet specifies trained CNN (grader 1 trained network - 14(x16) training images and 1(x16) validation image)
            DataSet = 'g1+cunefare';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
%             DataSet = '42';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
            ImageDir = [pname fname];
            startTimeCNN = clock
            [conelocs] = RunCNNnewSet_automatic(DataSet, ImageDir, boxposition, I);
            endTimeCNN = clock

            conelocs = unique(conelocs,'rows', 'stable'); 
            
%             conelocs(:,1) = p(1:2:end);
%             conelocs(:,2) = p(2:2:end);
            conelocs(:,3) = 1;
            
            conelocs(conelocs(:,1)<=minlimx,3)= 0;
            conelocs(conelocs(:,1)>=maxlimx,3)= 0;
            conelocs(conelocs(:,2)<=minlimy,3)= 0;
            conelocs(conelocs(:,2)>=maxlimy,3)= 0;
            
            conelocs = unique(conelocs, 'rows', 'stable');
            
            plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            conecount = size(conelocs(conelocs(:,3)~=0,1),1);
            totalconecount = size(conelocs,1);
            
            
        case 'No',
            conecount = 0;
    end % switch
    title([fname,'  ',num2str(conecount),' cones'],'Interpreter','none')
    
    if multiple_mosaics == 1
        change_mosaic{1} = I;
        mosaic = 1;
%         I = I(:,:,mosaic);
    else 
        change_mosaic{1} = I;
        mosaic = 1;
    end
    
elseif strcmp(ext,'.mat')==1
    load([pname fname]);
    conelocs = unique(conelocs, 'rows', 'stable');
    conecount = size(conelocs(conelocs(:,3)~=0,1),1);
    totalconecount = size(conelocs,1);
    
    minlimx = boxposition(1);
    minlimy = boxposition(2);
    maxlimx = minlimx + boxposition(3);
    maxlimy = minlimy + boxposition(4);
    
    if multiple_mosaics == 1
        change_mosaic = I;
        mosaic = 1;
        I = change_mosaic{1};
    else 
        change_mosaic = I;
        mosaic = 1;
    end
    
    Marker_Size = 4;
    
    h.fig = figure; imshow(I), title([fname,'  ',num2str(conecount),' cones'],'Interpreter','none'), hold on
    ax = gca;                    % get the current axis
    ttl = ax.Title;              % get the title text object
    ttl.FontSize = 14;
    ttl.BackgroundColor = [1 0 0];
        
    cla; a1 = imshow(I);title([fname,'  ',num2str(conecount),' cones'],'Interpreter','none'), hold on;
    rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
      
    
    plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
    
end


%% Main loop to wait for mouse clicks and keyboard
cont = 1;
while cont == 1
    [x,y,button] = ginput(1);
    
    % Mark cone center by left clicking
    if button == 1
%         x = round(x); y = round(y);       % rounded until 10.09.2019
%         x = x; y = y;                     % exact values from 11.09.2019 (also changes in next analysis steps)
                                            % CNN generates more exact cone
                                            % locations anyway
        
        if x<minlimx | x>maxlimx || y<minlimy | y>maxlimy
            conelocs(totalconecount+1,1) = x; conelocs(totalconecount+1,2) = y; conelocs(totalconecount+1,3) = 3;
        else
            conelocs(totalconecount+1,1) = x; conelocs(totalconecount+1,2) = y; conelocs(totalconecount+1,3) = 2;
        end
        
        conelocs = unique(conelocs, 'rows', 'stable');
        conecount = size(conelocs(conelocs(:,3)~=0,1),1);
        totalconecount = size(conelocs,1);
        
        plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
        % conecount = conecount+1;
        [Inr,  Num_mosaics] = size(change_mosaic);
        ttl.BackgroundColor = [1 0 0];
        title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
        
        
    end
    
    % Clear marked cone nearest to click location
    if ((button == 3) && size(conelocs,1)>0)
%         x = round(x); y = round(y);       % rounded until 10.09.2019 --> see button 1
        currentloc = [x y];
        %compute Euclidean distances:
        distances = sqrt(sum(bsxfun(@minus, conelocs(:,1:2), currentloc).^2,2));
        %find the smallest distance and delete that index in conelocs:
        conelocs(distances==min(distances),:)=[];
        
        conelocs = unique(conelocs, 'rows', 'stable');
        conecount = size(conelocs(conelocs(:,3)~=0,1),1);
        totalconecount = size(conelocs,1);
        
        % After deletion, plot current set of cones
        [Inr, Num_mosaics] = size(change_mosaic);
        cla; a1 = imshow(I);
        title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
        hold on;
        plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
             rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        
    end
    
    % Zoom in and out with two keys "+", "-" on numpad
    if button==45
        ax = axis; width=ax(2)-ax(1); height=ax(4)-ax(3);
        axis([ax(1)-width/3 ax(2)+width/3 ax(3)-height/3 ax(4)+height/3]);
        zoom(1/2);
    elseif button==43
        ax = axis; width=ax(2)-ax(1); height=ax(4)-ax(3);
        axis([x-width/2 x+width/2 y-height/2 y+height/2]);
        zoom(2);
    end
    
    
    % Pan image laterally with arrow keys
    if button==28 % LEFT arrow
        ax = axis;
        width=ax(2)-ax(1);
        height=ax(4)-ax(3);
        increment = round(width/20);
        axis([ax(1)-increment ax(2)-increment ax(3) ax(4)]);
        
    elseif button==29 % RIGHT arrow
        ax = axis;
        width=ax(2)-ax(1);
        height=ax(4)-ax(3);
        increment = round(width/20);
        axis([ax(1)+increment ax(2)+increment ax(3) ax(4)]);
        
    elseif button==30 % UP arrow
        ax = axis;
        width=ax(2)-ax(1);
        height=ax(4)-ax(3);
        increment = round(height/20);
        axis([ax(1) ax(2) ax(3)-increment ax(4)-increment]);
        
    elseif button==31 % DOWN arrow
        ax = axis;
        width=ax(2)-ax(1);
        height=ax(4)-ax(3);
        increment = round(height/20);
        axis([ax(1) ax(2) ax(3)+increment ax(4)+increment]);
        
    end
    
    
    % Darken image
    if button == 100  % 'D'
        if gamma<2
            gamma = gamma+0.1;
            J = imadjust(I,[],[],gamma);
            cla; a1 = imshow(J); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
        end
    end
    
    % Brighten image
    if button == 98  %'B'
        if gamma>0.1
            gamma = gamma - 0.1;
            J = imadjust(I,[],[],gamma);
            cla; a1 = imshow(J); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
        end
    end
    
    
    % Log contrast image
    if button == 108  %'L'
        if log_image
            I_INbox = log(double(I(minlimy+20:maxlimy-20,minlimx+20:maxlimx-20)));
            
            I_log = log(double(I));
            I_log = I_log - min(I_INbox(:));
            I_log = I_log./max(I_log(:));
            
            cla; a1 = imshow(I_log); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            log_image = 0;    
        elseif log_image == 0
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            log_image = 1;
        end
    end
    
    % Reset image gamma to default
    if button == 114  % 'R'
        gamma = 1.0;
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
    end
    
    % Toggle show marked locations
    if button == 116  % 'T'
        if markson
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            markson = 0;
        else
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            markson = 1;
        end
    end
 
    % Change Size of Box/Rectangle
    if button == 115  % 'S'
        change_box = questdlg('Select new Box parameters', 'Change Size of Box/Rectangle', 'Keep current box', ...
            'Select new 500x500p box', 'Enter new box-parameters', 'Keep current box')
        
        if strcmp(change_box, 'Keep current box')
            % do nothing
        elseif strcmp(change_box, 'Select new 500x500p box')
            old_boxposition = boxposition;
            hi = imrect(gca,[0 0 500 500]);
            boxposition = wait(hi);
            minlimx = boxposition(1);
            minlimy = boxposition(2);
            maxlimx = minlimx + boxposition(3);
            maxlimy = minlimy + boxposition(4);
            
            minlimx_old = old_boxposition(1);
            minlimy_old = old_boxposition(2);
            maxlimx_old = minlimx_old + old_boxposition(3);
            maxlimy_old = minlimy_old + old_boxposition(4);

            conelocs(conelocs(:,1)>minlimx & conelocs(:,1)<=minlimx_old,3)= 1;
            conelocs(conelocs(:,1)<maxlimx & conelocs(:,1)>=maxlimx_old,3)= 1;
            conelocs(conelocs(:,2)>minlimy & conelocs(:,2)<=minlimy_old,3)= 1;
            conelocs(conelocs(:,2)<maxlimy & conelocs(:,2)>=maxlimy_old,3)= 1;

            conelocs(conelocs(:,1)<=minlimx,3)= 0;
            conelocs(conelocs(:,1)>=maxlimx,3)= 0;
            conelocs(conelocs(:,2)<=minlimy,3)= 0;
            conelocs(conelocs(:,2)>=maxlimy,3)= 0;
            
            conelocs = unique(conelocs, 'rows', 'stable');
            conecount = size(conelocs(conelocs(:,3)~=0,1),1);
            totalconecount = size(conelocs,1);
            
            cla; a1 = imshow(I); hold on;
            ax = gca; ttl = ax.Title; ttl.FontSize = 14;
            ttl.BackgroundColor = [1 0 0];
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            [Inr,  Num_mosaics] = size(change_mosaic);
            title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
            
        elseif strcmp(change_box, 'Enter new box-parameters')
            old_boxposition = boxposition;
            boxposition = input('New box-parameters ([x y x-range y-range]: ');
            
            minlimx = boxposition(1);
            minlimy = boxposition(2);
            maxlimx = minlimx + boxposition(3);
            maxlimy = minlimy + boxposition(4);
            
            minlimx_old = old_boxposition(1);
            minlimy_old = old_boxposition(2);
            maxlimx_old = minlimx_old + old_boxposition(3);
            maxlimy_old = minlimy_old + old_boxposition(4);

            conelocs(conelocs(:,1)>minlimx & conelocs(:,1)<=minlimx_old,3)= 1;
            conelocs(conelocs(:,1)<maxlimx & conelocs(:,1)>=maxlimx_old,3)= 1;
            conelocs(conelocs(:,2)>minlimy & conelocs(:,2)<=minlimy_old,3)= 1;
            conelocs(conelocs(:,2)<maxlimy & conelocs(:,2)>=maxlimy_old,3)= 1;

            conelocs(conelocs(:,1)<=minlimx,3)= 0;
            conelocs(conelocs(:,1)>=maxlimx,3)= 0;
            conelocs(conelocs(:,2)<=minlimy,3)= 0;
            conelocs(conelocs(:,2)>=maxlimy,3)= 0;
            
            conelocs = unique(conelocs, 'rows', 'stable');
            conecount = size(conelocs(conelocs(:,3)~=0,1),1);
            totalconecount = size(conelocs,1);
            
            cla; a1 = imshow(I); hold on;
            ax = gca; ttl = ax.Title; ttl.FontSize = 14;
            ttl.BackgroundColor = [1 0 0];
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
            [Inr,  Num_mosaics] = size(change_mosaic);
            title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
            
        end
            
    end
    
    % Toggle show image
    if button == 105  % 'I'
        blank = zeros(size(I));
        if imageon
            cla; a1 = imshow(blank); hold on
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            imageon = 0;
        else
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            imageon = 1;
        end
    end
    
    % Toggle show image and plot circular cone markings
    if button == 99  % 'C'
        blank = zeros(size(I));
        if imageon
            cla; a1 = imshow(blank); hold on
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'oy', 'MarkerSize', Marker_Size+3)
   
            imageon = 0;
        else
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            imageon = 1;
        end
    end

    % Toggle show image and plot hexagonal cone markings
    if button == 120  % 'x'
        blank = zeros(size(I));
        if imageon
            cla; a1 = imshow(blank); hold on
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'xy', 'MarkerSize', Marker_Size+3)
   
            imageon = 0;
        else
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            imageon = 1;
        end
    end
    
    
    
    % display only automatic found cones
    if button ==49 %'1'
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        autocones = conelocs(conelocs(:,3)==1,1:2);
        plot(autocones(:,1),autocones(:,2),'.m')
    end
    
    % display only user defined cones
    if button ==50 %'2'
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        usercones = conelocs(conelocs(:,3)==2,1:2);
        plot(usercones(:,1),usercones(:,2),'.y')
    end
    
    % toggle between both groups separate and united view
    if button ==48 %'0'
        if bothcones
            cla; a1 = imshow(I); hold on;
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            usercones = conelocs(conelocs(:,3)==2,1:2);
            autocones = conelocs(conelocs(:,3)==1,1:2);
            plot(autocones(:,1),autocones(:,2),'.m')
            plot(usercones(:,1),usercones(:,2),'.y')
            bothcones = 0;
        else
            cla; a1 = imshow(I); hold on;
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            bothcones = 1;
        end
    end
    
    
    % toggle Voronoi plot
    if button==118 %'V'
                    cla; a1 = imshow(I); hold on;
            voronoi_type = questdlg('Select coloring for Voronoi patches', 'Coloring Voronois', 'Cone Area', ...
                    'Number of Neighbors', 'Cone Area');
                tic;
        numcolors = 25;  % ********* change from 15 
%         max_area_pixel = 150;        % basis for dividing cone Area through numcolors
        max_area_pixel = 80;        % basis for dividing cone Area through numcolors
        corr_factor = max_area_pixel/numcolors; % to represent whole colormap over values
        idx = round(linspace(1,256,numcolors));
        neighcolors = map(idx,:);
        neighcolors = flipud(neighcolors);
        
        min_neighbors = 4;
        max_neighbors = 9;
        numcolorsNN = max_neighbors-min_neighbors;  % ********* change from 15 
%         corr_factorNN = numcolorsNN/max_neighbours; % to represent whole colormap over values
        idxNN = round(linspace(1,256,numcolorsNN));
        neighcolorsNN = map(idxNN,:);
        neighcolorsNN = flipud(neighcolorsNN);
        
        if voronoiplot

            % Calculate num neighbours from Voronoi diagram
            if size(conelocs,1)>3
                
                vorocones = conelocs(conelocs(:,3)~=0,1:2);
                vorocones = unique(vorocones, 'rows', 'stable');
                
                dt = delaunayTriangulation(vorocones(:,1),vorocones(:,2));
                [V,C] = voronoiDiagram(dt);
                [dump,G] = voronoin(vorocones(:,1:2));
                for j = 1 : length ( G )
                    NumNeighbors(j,1) = size(G{j},2);
                    ConeArea(j,1) = polyarea(V(C{j},1),V(C{j},2));
                    idxMinMax = round(linspace(min(ConeArea),max_area_pixel,numcolors));
                end
            end
            for j = 1:length(C)
                if all(C{j}~=1)
                    
                    if strcmp(voronoi_type, 'Number of Neighbors')  % VARIANTE 1: Falschfarben sind Anzahl Nachbarn
                        if NumNeighbors(j)<= min_neighbors
                            h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(1,:));
                        elseif NumNeighbors(j)> max_neighbors
                            h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(end,:));
                        elseif NumNeighbors(j)> min_neighbors
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(NumNeighbors(j),:),'FaceAlpha',0.1);
                            h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(NumNeighbors(j)-min_neighbors,:));
                        end
    
                    elseif strcmp(voronoi_type, 'Cone Area')        % VARIANTE 2: Falschfarben sind Patchgröße
                        if ConeArea(j)<=max_area_pixel
                            [val,pos]=min(abs(idxMinMax-ConeArea(j)));
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(round(ConeArea(j)/corr_factor),:));
                            h_p = patch(V(C{j},1),V(C{j},2),neighcolors(pos,:));
%                         if ConeArea(j)<=64
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(round(ConeArea(j)),:));
                        else
                            h_p = patch(V(C{j},1),V(C{j},2),neighcolors(end,:));
                        end
                    end                                             % VARIANTEN Ende
                end
            end
            voronoiplot=0;
        else
            cla; a1 = imshow(I); hold on;
               plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
   
            rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
            voronoiplot=1;
        end
        toc;
    end
    
    
    % toggle cone mosaic image (if multiple images were selected)
    if button==109 %'M'
        if multiple_mosaics
            [Inr,  Num_mosaics] = size(change_mosaic);
            if mosaic < Num_mosaics
                mosaic = mosaic+1;
                I = change_mosaic{mosaic};
                
                cla; a1 = imshow(I); hold on;
                rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
                title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
                
            elseif mosaic == Num_mosaics
                mosaic = 1;
                I = change_mosaic{mosaic};
                
                cla; a1 = imshow(I); hold on;
                rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
                plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
                title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones', '  mosaic: ', num2str(mosaic), '/', num2str(Num_mosaics)],'Interpreter','none')
            end
        end
    end
    
    % remove conelocs in selected part (box) of the image
    if button==127 %'entf'

                delBox = imrect(gca);
        box_delCones = wait(delBox);
        
        mindelx = box_delCones(1);
        mindely = box_delCones(2);
        maxdelx = mindelx + box_delCones(3);
        maxdely = mindely + box_delCones(4);
        
        conelocs((conelocs(:,1)>=mindelx & conelocs(:,1)<=maxdelx & conelocs(:,2)>=mindely & conelocs(:,2)<=maxdely),:)= [];
%         conelocs(conelocs(:,1)<=maxdelx,:)= [];
%         conelocs(conelocs(:,2)>=mindely & conelocs(:,2)<=maxdely,:)= [];
%         conelocs(conelocs(:,2)<=maxdely,:)= [];
        
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
        title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none')
    end
    
    
    % add conelocs found by previously used algorithm outside box
    if button==102 %'F'
        add_conelocs = [];

                fPFBox = imrect(gca);
        box_addCones = wait(fPFBox);
        
        minaddx = box_addCones(1);
        minaddy = box_addCones(2);
        maxaddx = minaddx + box_addCones(3);
        maxaddy = minaddy + box_addCones(4);
        
%         conelocs(conelocs(:,1)>=minaddx&conelocs(:,1)<=maxaddx&conelocs(:,2)>=minaddy&conelocs(:,2)<=maxaddy,3)= 3;
%         conelocs(conelocs(:,1)<=maxaddx,3)= 3;
%         conelocs(conelocs(:,2)>=minaddy,3)= 3;
%         conelocs(conelocs(:,2)<=maxaddy,3)= 3;
        
        
        p = FastPeakFind(I(minaddy:maxaddy,minaddx:maxaddx));
        add_conelocs(:,1) = p(1:2:end)+minaddx-1;
        add_conelocs(:,2) = p(2:2:end)+minaddy-1;
        add_conelocs(:,3) = 3;
        
        conelocs = [conelocs; add_conelocs];
        
        conelocs = unique(conelocs, 'rows', 'stable');
        
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
        title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none')
        
%         clear add_conelocs
    end
    

    % add conelocs in selected part (box) of the image
    if button==97 %'A'

                fPFBox = imrect(gca);
        box_addCones = wait(fPFBox);
        
        minaddx = box_addCones(1);
        minaddy = box_addCones(2);
        maxaddx = minaddx + box_addCones(3);
        maxaddy = minaddy + box_addCones(4);
        
        conelocs(conelocs(:,1)>=minaddx&conelocs(:,1)<=maxaddx&conelocs(:,2)>=minaddy&conelocs(:,2)<=maxaddy,3)= 3;
%         conelocs(conelocs(:,1)<=maxaddx,3)= 3;
%         conelocs(conelocs(:,2)>=minaddy,3)= 3;
%         conelocs(conelocs(:,2)<=maxaddy,3)= 3;
        
%         
%             DataSet = 'g1+cunefare';  % dataSet specifies trained CNN (grader 1 14(x16)training + 100 cunefare val images and 1(x16) validation g1 + 16 cunefare training images)
%             ImageDir = [pname fname];
%             [add_conelocs] = RunCNNnewSet_automatic(DataSet, ImageDir, boxposition, I);
%             add_conelocs(:,3) = 3;
        
%         p = FastPeakFind(I(minaddy:maxaddy,minaddx:maxaddx));
%         add_conelocs(:,1) = p(1:2:end)+minaddx-1;
%         add_conelocs(:,2) = p(2:2:end)+minaddy-1;
%         add_conelocs(:,3) = 3;
        
%         conelocs = [conelocs; add_conelocs];
        
        conelocs = unique(conelocs, 'rows', 'stable');
        
        cla; a1 = imshow(I); hold on;
        rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
        plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
        title([fname,'  ',num2str(size(conelocs(conelocs(:,3)~=0,1),1)),' cones'],'Interpreter','none')
        
%         clear add_conelocs
    end
    
    
    
    
    % Spacebar to save current locations to mat file
    if button==32  %'SPACEBAR'
        if multiple_mosaics
            I = change_mosaic;
        end
        if strcmp(ext,'.tiff')==1 || strcmp(ext,'.tif')==1
            name1 = [pstr filesep nm '_annotated.png'];
            name2 = [pstr filesep nm '_annotated.mat'];
        elseif strcmp(ext,'.mat')==1
            name1 = [pstr filesep nm '.png'];
            name2 = [pname fname];
        end
        h.figprint = gcf; print(h.figprint,name1, '-dpng');
        
        save(name2, 'conecount', 'I', 'conelocs','boxposition','multiple_mosaics');
        ttl.BackgroundColor = [0 1 0];
        
        if multiple_mosaics
            I = change_mosaic;
        end
        
    end
    
    % Escape key to quit count routine
    if button==27
        close_dialog = questdlg('Do you want to save before quitting?',...
            'Close analysis window', 'Yes', 'No', 'Yes');
        if strcmp(close_dialog, 'Yes')
            % copied from SPACEBAR input
            if multiple_mosaics
                I = change_mosaic;
            end
            if strcmp(ext,'.tiff')==1 || strcmp(ext,'.tif')==1
                name1 = [pstr filesep nm '_annotated.png'];
                name2 = [pstr filesep nm '_annotated.mat'];
            elseif strcmp(ext,'.mat')==1
                name1 = [pstr filesep nm '.png'];
                name2 = [pname fname];
            end
            h.figprint = gcf; print(h.figprint,name1, '-dpng');
%             saveas(gcf,[name1(1:end-4),'.epsc'])
            save(name2, 'conecount', 'I', 'conelocs','boxposition','multiple_mosaics');
            ttl.BackgroundColor = [0 1 0];

            if multiple_mosaics
                I = change_mosaic;
            end
        end
        
        cont = 0;
        close(h.fig);
        return
    end
end


% FIG = figure; imshow(I), title([fname,'  ',num2str(conecount),' cones'],'Interpreter','none'), hold on
% ax = gca;                    % get the current axis
% ttl = ax.Title;              % get the title text object
% ttl.FontSize = 14;
% ttl.BackgroundColor = [1 0 0];
% 
% cla; a1 = imshow(I);title([fname,'  ',num2str(conecount),' cones'],'Interpreter','none'), hold on;
% rectangle('Position',boxposition,'EdgeColor',[1 0 1]);
% plot(conelocs(conelocs(:,3)~=0,1),conelocs(conelocs(:,3)~=0,2),'.y', 'MarkerSize', Marker_Size)
% 
% 
% voronoi_type = 'Cone Area';
%             for j = 1:length(C)
%                 if all(C{j}~=1)
%                     
%                     if strcmp(voronoi_type, 'Number of Neighbors')  % VARIANTE 1: Falschfarben sind Anzahl Nachbarn
%                         if NumNeighbors(j)<= min_neighbors
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(1,:));
%                         elseif NumNeighbors(j)> max_neighbors
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(end,:));
%                         elseif NumNeighbors(j)> min_neighbors
% %                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(NumNeighbors(j),:),'FaceAlpha',0.1);
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolorsNN(NumNeighbors(j)-min_neighbors,:));
%                         end
%     
%                     elseif strcmp(voronoi_type, 'Cone Area')        % VARIANTE 2: Falschfarben sind Patchgröße
%                         if ConeArea(j)<=max_area_pixel
%                             [val,pos]=min(abs(idxMinMax-ConeArea(j)));
% %                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(round(ConeArea(j)/corr_factor),:));
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(pos,:));
% %                         if ConeArea(j)<=64
% %                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(round(ConeArea(j)),:));
%                         else
%                             h_p = patch(V(C{j},1),V(C{j},2),neighcolors(end,:));
%                         end
%                     end                                             % VARIANTEN Ende
%                 end
%             end
% 
% 
% nameForSaving = [pstr filesep nm '_annotated'];
% hgexport(FIG, [nameForSaving '_2016Voronoi.tiff'], hgexport('factorystyle'), 'Format', 'tiff')
% hgexport(FIG, [nameForSaving '_2016Voronoi.epsc'], hgexport('factorystyle'), 'Format', 'epsc')

