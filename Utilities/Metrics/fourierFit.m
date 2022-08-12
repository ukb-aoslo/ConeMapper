function [spacingInd, predictions, err, fitParams] = fourierFit(fourierProfile, prior, showPlots)
%FUNCTION [spacingInd, predictions, err, fitParams] = fourierFit(fourierProfile, prior, showPlots)
%   Robert Cooper
%
    if nargin < 3
        showPlots = false;
    end
    
    % Set up initial guess for fit parameters -----------------

    % Remove any nan and inf.
    fourierProfile = fourierProfile(~isnan(fourierProfile));
    fourierProfile = fourierProfile(~isinf(fourierProfile));
    fourierProfile = fourierProfile - min(fourierProfile);

    fitSampling = 0.25;

    timeBase = 1:(length(fourierProfile));
    fineTimeBase = 1:fitSampling:(length(fourierProfile));

    fourierSampling = (timeBase / (size(fourierProfile,2)*2) );
    fineFourierSampling = (fineTimeBase / (size(fourierProfile,2)*2) );

    % Start plot
    if showPlots
        thePlot = figure(1); 
        clf; 
        hold on
        set(gca,'FontName','Helvetica','FontSize',14);
        plot(fourierSampling, fourierProfile,'k');
    end

    if isempty(prior)
        [initshift, ~, initshiftind] = fourierFit_rough(fourierProfile, showPlots);

        fitParams.shift = initshift;
        % Make initial guesses
        fitParams.scale1 = fourierProfile(1) - fourierProfile(initshiftind);
        fitParams.decay1 = (fourierProfile(1)*0.36) / (fitParams.shift);

        fitParams.exp1 = exp(1);
        [maxval, ~] = max(fourierProfile);
        maxval = maxval+1;

        fitParams.offset1 = maxval - fitParams.scale1;
        fitParams.scale2 =  fitParams.offset1 * 0.3679;
        fitParams.decay2 = (max(fourierSampling)-fitParams.shift) / (fitParams.shift*0.36);
        fitParams.exp2 = exp(1);

    else
        fitParams = prior;
    end

    % Add initial guess to the plot
    predictions0 = ComputeModelPreds(fitParams, fourierSampling);
    if showPlots
        figure(thePlot); 
        hold on; 
        plot(fourierSampling, predictions0, 'k', 'LineWidth', 2); 
        hold off;
    end

    % Fit ------------------------------

    % Set fmincon options
    options = optimset('fmincon');
    options = optimset(options,...
        'Diagnostics', 'off',...
        'Display', 'off',...
        'LargeScale', 'off',...
        'Algorithm', 'sqp');

    x1 = ParamsToX(fitParams);
    % [scale1 decay1 offset1 exp1 scale2 decay2 exp2 shift]
    vlb = [0.5, 0.001, 0.01, 1, 0.001, 0.001, 1, initshift-0.1];
    vub = [5, 25, 15, 10, 15, 25, 10, initshift+0.1];

    x = fmincon(@(x)FitModelErrorFunction(x,fourierSampling,fourierProfile,fitParams),...
        x1, [], [], [], [], vlb, vub, [], options);

    % Extract fit parameters
    fitParams = XToParams(x,fitParams);

    % Add final fit to plot
    predictions = ComputeModelPreds(fitParams, fourierSampling);

    if showPlots
        figure(thePlot); 
        hold on;
        plot(fourierSampling, predictions, 'g', 'LineWidth', 2);
        axis([0, max(fourierSampling), 0, 7]);
    end


    residuals = fourierProfile - predictions;
    spacing_val = fitParams.shift;
    spacingInd = find(fourierSampling <= spacing_val, 1, 'last' );

    fitops = fitoptions(...
        'Method', 'SmoothingSpline',...
        'SmoothingParam', 0.99995,...
        'Normalize','on');

    f = fit((1:length(residuals))', ...
        residuals',...
        'SmoothingSpline', fitops);

    if showPlots
        figure(2);
        clf; 
        plot(fourierSampling, residuals); 
        hold on; 
        plot(fineFourierSampling, f(1:fitSampling:length(residuals))');
        plot(spacing_val, residuals(spacingInd),'b*'); 
    end

    % Update the spacing index to be on the same scale as our sampling
    spacingInd = spacingInd / fitSampling;

    residuals = f(1:fitSampling:length(residuals))';
    preval = residuals(spacingInd-1) - residuals(spacingInd);

    % Find our closest peak -------------------------------

    % Set the minbound to the bottom of the first downsweep, or 10 indexes in
    % (impossible spacing to us to detect anyway)
    diffresiduals = diff(residuals);

    if diffresiduals(1) >= 0
        minbound = 10 / fitSampling;
    else
        for indI = 1:length(diffresiduals)
            if diffresiduals(indI) >= 0 
                minbound = indI;
                break;
            end
        end
    end

    maxbound = length(fourierProfile) - 2;

    platstart = NaN;
    for indI = spacingInd-1:-1:minbound
        thisval = residuals(indI-1)-residuals(indI);

        % If we're on a plateau, track it.
        if thisval <= eps && thisval >= -eps
            %The plateau would've started before this index if thisval is 0.
            platstart = indI; 
        end

        % It should only be increasing or flat- if it isn't anymore and heads down, kick out.
        if preval >= 0 && thisval >= 0 
            spacingInd = indI; 
        elseif thisval < 0
            if isnan(platstart)
                spacingInd = indI;
            else
                spacingInd = (platstart+indI) / 2;
            end

            break;
        end

        % If thisval isn't 0 anymore, we're not on a plataeu.
        if thisval >= eps || thisval <= -eps
            platstart = NaN;
        end
        preval = thisval;
    end


    % Determine Sharpness of the peak as an error measurment ---------------
    flattenedSpacing = floor(spacingInd);
    lowFreqBound = flattenedSpacing;
    highFreqBound = flattenedSpacing;

    sharpResiduals = residuals;
    % Find our two closest peaks -------------------------------------------

    % Use a smoothed residual to find the bottoms of our peaks.
    for indI = (flattenedSpacing-1):-1:minbound 
        thisval = sharpResiduals(indI-1) - sharpResiduals(indI);

        lowFreqBound = indI; 
        if thisval > 0.01
            if showPlots
                figure(2); 
                hold on;
                plot(fineFourierSampling(lowFreqBound), residuals(lowFreqBound),'g*')
            end

            break;
        end
    end

    for indI = (flattenedSpacing+1):1:maxbound
        thisval = sharpResiduals(indI+1) - sharpResiduals(indI);

        highFreqBound = indI;
        if thisval > 0.01
            if showPlots
                figure(2); 
                hold on;
                plot(fineFourierSampling(highFreqBound), residuals(highFreqBound),'g*')
            end

            break;
        end
    end

    maxamplitude = max(residuals(minbound:maxbound)) - min(residuals(minbound:maxbound));

    if lowFreqBound == (flattenedSpacing-1)...
            && highFreqBound ~= flattenedSpacing

        highheight = (residuals(flattenedSpacing) - residuals(highFreqBound));
        heightdistinct = highheight ./ maxamplitude;

    elseif highFreqBound == (flattenedSpacing+1)...
            && lowFreqBound ~= flattenedSpacing

        lowheight = (residuals(flattenedSpacing) - residuals(lowFreqBound));
        heightdistinct = lowheight ./ maxamplitude;

    elseif highFreqBound ~=(flattenedSpacing+1)...
            && lowFreqBound ~= (flattenedSpacing-1)

        % Find the distinctness of our peak based on the average height of the two
        % sides of the triangle
        lowheight = residuals(flattenedSpacing) - residuals(lowFreqBound);
        highheight = residuals(flattenedSpacing) - residuals(highFreqBound);
        heightdistinct = max([lowheight highheight]) ./ maxamplitude;
    else
        heightdistinct = 0;
    end

    % Revert to our original sampling.
    spacingInd = spacingInd * fitSampling;
    err = heightdistinct;

    if showPlots
        figure(2);
        hold on; plot(fineFourierSampling(flattenedSpacing), residuals(flattenedSpacing),'r*');
        hold off;
        figure(1); 
        plot(fineFourierSampling(flattenedSpacing), fourierProfile(floor(flattenedSpacing*fitSampling)),'r*')
        title([' Confidence: ' num2str(err) ]);
        hold off;
        drawnow;
    end
end

%% Help functions

function f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)
% f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)
% Search error function

    % Extract parameters into meaningful structure
    fitParams = XToParams(x, fitParams);

    % Make predictions
    preds = ComputeModelPreds(fitParams, timeBase);

    % Compute fit error as RMSE
    nPoints = length(theResponse);
    theDiff2 = (theResponse-preds) .^ 2;
    f = 100 * sqrt(sum(theDiff2)/nPoints);
end

function x = ParamsToX(params)
% x = ParamsToX(params)
% Convert parameter structure to vector of parameters to search over
    x = [params.scale1, params.decay1, params.offset1, params.exp1, params.scale2, params.decay2, params.exp2, params.shift];
end

function params = XToParams(x,params)
% fitParams = XToParams(x,params)
% Convert search params and base structure to filled in structure.
    params.scale1 = x(1);
    params.decay1 = x(2);
    params.offset1 = x(3);
    params.exp1 = x(4);
    params.scale2 = x(5);
    params.decay2 = x(6);
    params.exp2 = x(7);
    params.shift = x(8);
end

function fullExp = ComputeModelPreds(params,freqBase)
% preds =  ComputeModelPreds(params,t)
% Compute the predictions of the model

    fullExp = params.offset1 + params.scale1*params.exp1.^( -params.decay1 * freqBase );
    
    bottomExpLoc = find(freqBase>params.shift);
    if isempty(bottomExpLoc)
        bottomExpLoc = 1;
    end
    bottomExpTime = freqBase(bottomExpLoc);

    % The exponential must always line up with the other exponential function's
    % value!   
    maxmatch = fullExp(bottomExpLoc(1)) - params.scale2;
    fullExp(bottomExpLoc) = maxmatch + params.scale2*params.exp2.^( -params.decay2 * (bottomExpTime-bottomExpTime(1)) );
end