function result = RemoveConesOusideImage(image, conelocs)
%REMOVECONESOUSIDEIMAGE Remove the cones which is not places on the actual image.
    sizeImage = size(image);
    
    % delete cones outside of the image canvas
    conelocs((conelocs(:,1) < 1) | (conelocs(:,2) < 1), :) = [];
    conelocs((conelocs(:,1) > sizeImage(2)) | (conelocs(:,2) > sizeImage(1)), :) = [];
    
    conelocsY = int64(conelocs(:,1));
    conelocsX = int64(conelocs(:,2));
    
    values = image(sub2ind(sizeImage, conelocsX', conelocsY'));
    result = conelocs(values > 2, :);
end

