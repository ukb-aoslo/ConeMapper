function [spacing_ind, predictions, err, fitParams] = fourierFit(fourierProfile, prior, doplots)

if ~exist('doplots')
    doplots = false;
end


%% Set up initial guess for fit parameters

% Remove any nan and inf.
fourierProfile = fourierProfile(~isnan(fourierProfile));
fourierProfile = fourierProfile(~isinf(fourierProfile));
fourierProfile = fourierProfile-min(fourierProfile);

fitSampling = 0.25;

timeBase = 1:(length(fourierProfile));
fineTimeBase = 1:fitSampling:(length(fourierProfile));

fourierSampling =(timeBase/(size(fourierProfile,2)*2));
fineFourierSampling =(fineTimeBase/(size(fourierProfile,2)*2));

%% Start plot
if doplots
    thePlot = figure(1); clf; hold on
    set(gca,'FontName','Helvetica','FontSize',14);
    plot(fourierSampling, fourierProfile,'k');
end

if isempty(prior)
    
    [initshift, firsterr, initshiftind] = fourierFit_rough(fourierProfile, doplots);
    
    
    
    fitParams.shift = initshift;
    % Make initial guesses
    fitParams.scale1 = fourierProfile(1)-fourierProfile(initshiftind);
    fitParams.decay1 = (fourierProfile(1)*.36) /...
                        (fitParams.shift);

    fitParams.exp1 = exp(1);
    [maxval, maxind] = max(fourierProfile);
%     if maxind ~= 1 % If the maximum value isn't the first index, 
                   % then ensure that the fit doesn't start touching the
                   % data
        maxval = maxval+1;
%     end
                    
    fitParams.offset1 = maxval-fitParams.scale1;
    fitParams.scale2 =  fitParams.offset1*.3679;
    fitParams.decay2 = (max(fourierSampling)-fitParams.shift)/ (fitParams.shift*.36);
    fitParams.exp2 = exp(1);
        
else
    fitParams = prior;
end

% Add initial guess to the plot
predictions0 = ComputeModelPreds(fitParams,fourierSampling);
if doplots
    figure(thePlot); hold on; plot(fourierSampling,predictions0,'k','LineWidth',2); hold off;
end

%% Fit
initParams = fitParams;
% Set fmincon options
options = optimset('fmincon');
options = optimset(options,'Diagnostics','off','Display','off','LargeScale','off','Algorithm','sqp');

x1 = ParamsToX(fitParams);
% [scale1 decay1 offset1 exp1 scale2 decay2 exp2 shift]
vlb = [0.5 0.001 0.01 1 0.001  0.001 1  initshift-0.1];
vub = [5 25   15   10 15     25       10  initshift+0.1];

x = fmincon(@(x)FitModelErrorFunction(x,fourierSampling,fourierProfile,fitParams),x1,[],[],[],[],vlb,vub,[],options);
% [warnmsg, msgid] = lastwarn;
% if ~isempty(warnmsg) 
%    disp('Wut'); 
% end
% Extract fit parameters
fitParams = XToParams(x,fitParams);

% Add final fit to plot
predictions = ComputeModelPreds(fitParams,fourierSampling);

if doplots
    figure(thePlot); hold on; plot(fourierSampling,predictions,'g','LineWidth',2);
    axis([0 max(fourierSampling) 0 7]);
end


residuals = fourierProfile-predictions;
spacing_val = fitParams.shift;
spacing_ind = max(find(fourierSampling<=spacing_val));

fitops = fitoptions('Method','SmoothingSpline','SmoothingParam',0.99995,'Normalize','on');
% residuals = medfilt1(residuals,7);
f = fit([1:length(residuals)]',residuals','SmoothingSpline',fitops);

% residuals = medfilt1(residuals,5);

if doplots
    figure(2); clf; plot(fourierSampling, residuals); hold on; plot(fineFourierSampling, f(1:fitSampling:length(residuals))');
    plot(spacing_val, residuals(spacing_ind),'b*'); 
end

% Update the spacing index to be on the same scale as our sampling
spacing_ind = spacing_ind/fitSampling;

residuals = f(1:fitSampling:length(residuals))';
preval = residuals(spacing_ind-1)-residuals(spacing_ind);

%% Find our closest peak

% Set the minbound to the bottom of the first downsweep, or 10 indexes in
% (impossible spacing to us to detect anyway)
diffresiduals = diff(residuals);

if diffresiduals(1) >= 0
    minbound = 10/fitSampling;
else
    for i=1:length(diffresiduals)
        if diffresiduals(i) >= 0 
            minbound = i;
            break;
        end
    end
end

maxbound = length(fourierProfile)-2;



platstart=NaN;
for i=spacing_ind-1:-1:minbound
   
    thisval = residuals(i-1)-residuals(i);
    
    % If we're on a plateau, track it.
    if thisval<=eps && thisval>=-eps
        platstart = i; %The plateau would've started before this index if thisval is 0.
    end
    
    if preval>=0 && thisval>=0 % It should only be increasing or flat- if it isn't anymore and heads down, kick out.
        spacing_ind=i; 

    elseif thisval<0 %&& ((residuals(i-1)>0) || (residuals(i)>0))
        if isnan(platstart)
            spacing_ind=i;
        else
            spacing_ind=(platstart+i)/2;
        end
        
        break;
    end
    
    % If thisval isn't 0 anymore, we're not on a plataeu.
    if thisval>=eps || thisval<=-eps
        platstart=NaN;
    end
    preval = thisval;
end


%% Determine Sharpness of the peak as an error measurment
flattened_spacing = floor(spacing_ind);
lowfreqbound=flattened_spacing;
highfreqbound=flattened_spacing;

sharpresiduals = residuals; %f(1:length(residuals))';
%% Find our two closest peaks

% Use a smoothed residual to find the bottoms of our peaks.
for i=(flattened_spacing-1):-1:minbound 
   
    thisval = sharpresiduals(i-1)-sharpresiduals(i);
    
    if thisval<=0.01 
        lowfreqbound=i; 

    elseif thisval>0.01
        lowfreqbound=i; 
        if doplots
            figure(2); hold on;
            plot(fineFourierSampling(lowfreqbound), residuals(lowfreqbound),'g*')
        end
        break;
    end
    preval = thisval;
end
%%
for i=(flattened_spacing+1):1:maxbound
   
    thisval = sharpresiduals(i+1)-sharpresiduals(i);
    
    if thisval<=0.01 
        highfreqbound=i; 

    elseif thisval>0.01
        highfreqbound=i;
        if doplots
            figure(2); hold on;
            plot(fineFourierSampling(highfreqbound), residuals(highfreqbound),'g*')
        end
        break;
    end
    preval = thisval;
end

maxamplitude = max(residuals(minbound:maxbound))-min(residuals(minbound:maxbound));

if lowfreqbound==(flattened_spacing-1) && highfreqbound~=flattened_spacing
    
    highheight = (residuals(flattened_spacing) - residuals(highfreqbound));

    heightdistinct = highheight./maxamplitude;
    
elseif highfreqbound==(flattened_spacing+1) && lowfreqbound~=flattened_spacing
    
    lowheight = (residuals(flattened_spacing) - residuals(lowfreqbound));

    heightdistinct = lowheight./maxamplitude;
    
elseif highfreqbound~=(flattened_spacing+1) && lowfreqbound~=(flattened_spacing-1)
    % Find the distinctness of our peak based on the average height of the two
    % sides of the triangle
    lowheight = residuals(flattened_spacing) - residuals(lowfreqbound);
    highheight = residuals(flattened_spacing) - residuals(highfreqbound);

    heightdistinct = max([lowheight highheight])./maxamplitude;
else
    heightdistinct=0;
end


% Revert to our original sampling.
spacing_ind = spacing_ind*fitSampling;
err = heightdistinct;

if doplots

    figure(2);
    hold on; plot(fineFourierSampling(flattened_spacing), residuals(flattened_spacing),'r*');
    hold off;
    figure(1); 
    plot(fineFourierSampling(flattened_spacing), fourierProfile(floor(flattened_spacing*fitSampling)),'r*')
    title([' Confidence: ' num2str(err) ]);
        hold off;
    drawnow;
%     pause;
end



end

% f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)
%
% Search error function
function f = FitModelErrorFunction(x,timeBase,theResponse,fitParams)

% Extract parameters into meaningful structure
fitParams = XToParams(x,fitParams);

% Make predictions
preds = ComputeModelPreds(fitParams,timeBase);

% Compute fit error as RMSE
nPoints = length(theResponse);
theDiff2 = (theResponse-preds).^2;
f = 100*sqrt(sum(theDiff2)/nPoints);
% figure(333); hold on; plot(f,'.'); hold off;
end

% x = ParamsToX(params)
%
% Convert parameter structure to vector of parameters to search over
function x = ParamsToX(params)
    x = [params.scale1 params.decay1 params.offset1 params.exp1 params.scale2 params.decay2 params.exp2 params.shift];
end


% fitParams = XToParams(x,params)
%
% Convert search params and base structure to filled in structure.
function params = XToParams(x,params)
params.scale1 = x(1);
params.decay1 = x(2);
params.offset1 = x(3);
params.exp1 = x(4);
params.scale2 = x(5);
params.decay2 = x(6);
params.exp2 = x(7);
params.shift = x(8);
end

% preds =  ComputeModelPreds(params,t)
%
% Compute the predictions of the model
function fullExp = ComputeModelPreds(params,freqBase)

fullExp = params.offset1 + params.scale1*params.exp1.^( -params.decay1 * freqBase );


bottomExpLoc = find(freqBase>params.shift);
if isempty(bottomExpLoc)
    bottomExpLoc = 1
end
bottomExpTime = freqBase(bottomExpLoc);

% The exponential must always line up with the other exponential function's
% value!   
maxmatch = fullExp(bottomExpLoc(1))-params.scale2;

fullExp(bottomExpLoc) = maxmatch + params.scale2*params.exp2.^( -params.decay2 * (bottomExpTime-bottomExpTime(1)) );

end
