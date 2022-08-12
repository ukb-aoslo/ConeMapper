function [shift, err, shiftInd] = fourierFit_rough(fourierProfile, showPlots)
%FUNCTION [shift, err, shiftInd] = fourierFit_rough(fourierProfile, showPlots)
%   Robert Cooper
%
    if nargin < 2
        showPlots = false;
    end

    % Set up initial guess for fit parameters ---------------
    % Remove any nan and inf.
    fourierProfile = fourierProfile(~isnan(fourierProfile));
    fourierProfile = fourierProfile(~isinf(fourierProfile));
    fourierProfile = fourierProfile-min(fourierProfile);

    timeBase = 1:length(fourierProfile);
    fourierSampling = (timeBase / (size(fourierProfile,2)*2) );

    % Plot
    if showPlots
        thePlot = figure(10); 
        clf; 
        hold on
        set(gca,'FontName','Helvetica','FontSize',14);
        plot(fourierSampling,fourierProfile,'k'); 
        axis([0 max(fourierSampling) 0 7])
    end

    % Make initial guesses    
    fitParams.scale1 = max(fourierProfile)*0.9-min(fourierProfile);
    fitParams.decay1 = 1;
    fitParams.offset1 = 0;
    fitParams.exp1 = exp(1);
    fitParams.shift = 0;

    % Add initial guess to the plot
    predictions0 = ComputeModelPreds(fitParams,fourierSampling);

    if showPlots
        figure(thePlot); 
        hold on; 
        plot(fourierSampling, predictions0, 'k', 'LineWidth', 2); 
        hold off;
    end


    % Fit ----------------------

    % Set fmincon options
    options = optimset('fmincon');
    options = optimset(options, ...
        'Diagnostics', 'off',...
        'Display', 'off',...
        'LargeScale', 'off',...
        'Algorithm', 'interior-point');

    x1 = ParamsToX(fitParams);

    % scale decay offset shift
    vlb = [0.01, 0.001, -10, 1, 0];
    vub = [15, 15, 10, 10, max(fourierSampling)];

    [x, ~, ~] = fmincon(@(x)FitModelErrorFunction(x, fourierSampling, fourierProfile, fitParams),...
        x1, [], [], [], [], vlb, vub, [], options);

    % Extract fit parameters
    fitParams = XToParams(x,fitParams);

    % Add final fit to plot
    predictions = ComputeModelPreds(fitParams,fourierSampling);

    if showPlots
        figure(thePlot); 
        hold on; 
        plot(fourierSampling, predictions, 'g', 'LineWidth', 2); 
        hold off;
        drawnow;
    end

    % Find the second zero crossing (where the fit intersects with the curve)
    residuals = predictions - fourierProfile;

    fitops = fitoptions(...
        'Method', 'SmoothingSpline',...
        'SmoothingParam', 0.9999,...
        'Normalize', 'on');

    f = fit((1:length(residuals))',...
        residuals',...
        'SmoothingSpline', fitops);

    residuals = f(1:length(residuals))';

    [pks, locs] = findpeaks(fliplr(residuals*(-1)));
    [~, indL] = min(pks);
    maxNegativeDiffInd = locs(indL);

    % Find all local minima that are below 0 (the fit is underneath the data)
    locs = locs(pks > 0); 

    % Find the furthest out index of this peak.
    locs = length(fourierProfile)+1-locs; 
    % Make sure it's not at the end- we won't be finding rods.
    locs = locs(locs < floor(2*length(fourierProfile)/3)); 
    % Make sure it's not at the beginning- we won't be finding blood vessels.
    locs = locs(locs > 6); 

    curveHeight = 1;

    prevals = residuals(1:3);
    % For each of the minima underneath the data
    for indL = 1:length(locs) 
        for indI = locs(indL):length(residuals)-1
            thisval = residuals(indI);

            % Find the zero crossings
            if all(prevals < 0) && (thisval > 0)
                break;
            end

            prevals(1:2) = prevals(2:3);
            prevals(3) = thisval;
        end

        % If the zero crossing was preceded by a lower minima,
        % then take/keep that as our starting point for the next step.
        if residuals(locs(indL)) < curveHeight
            curveHeight = residuals(locs(indL));
            maxNegativeDiffInd = locs(indL);
        end
    end

    if showPlots
        figure(11); 
        clf;
        hold on; 
        plot(fourierSampling(maxNegativeDiffInd), residuals(maxNegativeDiffInd), 'b*');
    end

    % Trace back to where it is maximally different from our fit.
    preval = residuals(maxNegativeDiffInd-1) - residuals(maxNegativeDiffInd);
    if round(preval, 5) < 0
        for indI = maxNegativeDiffInd-1:-1:2
            thisval = residuals(indI-1) - residuals(indI);

            % It should only be decreasing or flat- if it isn't anymore and heads upward, kick out.
            if round(preval, 5) <= 0 && round(thisval,5) <= 0 
                maxNegativeDiffInd=indI; 
            elseif thisval > 0.03
                break;
            end

            preval = thisval;
        end
    end

    if showPlots
        figure(thePlot);
        hold on;
        plot( fourierSampling(maxNegativeDiffInd), fourierProfile(maxNegativeDiffInd), 'r*' );
        hold off;
        figure(11);
        plot(fourierSampling, residuals );
        plot( fourierSampling(maxNegativeDiffInd), residuals(maxNegativeDiffInd), 'r*' );
        hold off;
    end

    err = fourierProfile(1);

    shift = fourierSampling(maxNegativeDiffInd + 1);
    shiftInd = maxNegativeDiffInd + 1;
end

%% Help functions

function f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)
% f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)
% Search error function

    % Extract parameters into meaningful structure
    fitParams = XToParams(x,fitParams);

    % Make predictions
    preds = ComputeModelPreds(fitParams,timeBase);

    % Compute fit error as RMSE
    nPoints = length(theResponse);
    theDiff2 = (theResponse-preds) .^ 2;
    f = 100 * sqrt(sum(theDiff2) / nPoints);
end

function x = ParamsToX(params)
% x = ParamsToX(params)
% Convert parameter structure to vector of parameters to search over

    x = [params.scale1, params.decay1, params.offset1, params.exp1, params.shift];
end

function params = XToParams(x,params)
% fitParams = XToParams(x,params)
% Convert search params and base structure to filled in structure.
    params.scale1 = x(1);
    params.decay1 = x(2);
    params.offset1 = x(3);
    params.exp1 = x(4);
    params.shift = x(5);
end

function fullExp = ComputeModelPreds(params,freqBase)
% preds =  ComputeModelPreds(params,t)
% Compute the predictions of the model
    
    fullExp = params.offset1 + params.scale1*params.exp1.^( -params.decay1 * (freqBase-params.shift) );
end
