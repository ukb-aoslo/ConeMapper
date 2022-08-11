function [shift, err, shiftind ] = fourierFit_rough(fourierProfile, doplots)

if ~exist('doplots')
    doplots = false;
end

%% Set up initial guess for fit parameters

% Remove any nan and inf.
fourierProfile = fourierProfile(~isnan(fourierProfile));
fourierProfile = fourierProfile(~isinf(fourierProfile));
fourierProfile = fourierProfile-min(fourierProfile);

timeBase = 1:length(fourierProfile);
fourierSampling =(timeBase/(size(fourierProfile,2)*2));

% Plot
if doplots
    thePlot = figure(10); clf; hold on
    set(gca,'FontName','Helvetica','FontSize',14);
    plot(fourierSampling,fourierProfile,'k'); axis([0 max(fourierSampling) 0 7])
end


% Make initial guesses    
fitParams.scale1 = max(fourierProfile)*0.9-min(fourierProfile);
fitParams.decay1 = 1;
fitParams.offset1 = 0;
fitParams.exp1 = exp(1);
fitParams.shift = 0;


% Add initial guess to the plot
predictions0 = ComputeModelPreds(fitParams,fourierSampling);
if doplots
    figure(thePlot); hold on; plot(fourierSampling,predictions0,'k','LineWidth',2); hold off;
end
%% Fit

% Set fmincon options
options = optimset('fmincon');
options = optimset(options,'Diagnostics','off','Display','off','LargeScale','off','Algorithm','interior-point');

x1 = ParamsToX(fitParams);
% fitParams;
% scale decay offset shift
vlb = [0.01 0.001  -10 1  0];
vub = [15   15     10  10 max(fourierSampling)];

[x,ffval,exitflag] = fmincon(@(x)FitModelErrorFunction(x,fourierSampling,fourierProfile,fitParams),x1,[],[],[],[],vlb,vub,[],options);

% Extract fit parameters
fitParams = XToParams(x,fitParams);

% Add final fit to plot
predictions = ComputeModelPreds(fitParams,fourierSampling);

% spacing = fitParams.shift;
if doplots
    figure(thePlot); hold on; plot(fourierSampling,predictions,'g','LineWidth',2); 
    hold off;drawnow;
end

% Find the second zero crossing (where the fit intersects with the curve)
residuals = predictions-fourierProfile;

fitops = fitoptions('Method','SmoothingSpline','SmoothingParam',.9999,'Normalize','on');
% residuals = medfilt1(residuals,7);
f = fit([1:length(residuals)]',residuals','SmoothingSpline',fitops);

residuals = f(1:length(residuals))';

[pks,locs] = findpeaks(fliplr(residuals*-1));
[~, l] = min(pks);
maxnegdiff_ind = locs(l);

locs = locs(pks>0); % Find all local minima that are below 0 (the fit is underneath the data)

locs = length(fourierProfile)+1-locs; % Find the furthest out index of this peak.
locs = locs(locs < floor(2*length(fourierProfile)/3)); % Make sure it's not at the end- we won't be finding rods.
locs = locs(locs > 6); % Make sure it's not at the beginning- we won't be finding blood vessels.

curheight = 1;
curind=1;

prevals = residuals(1:3);
for l=1:length(locs) % For each of the minima underneath the data,
        
    for i=locs(l):length(residuals)-1

        thisval = residuals(i);

        if all(prevals<0) && thisval>0 % Find the zero crossings
            curind=i;
            break;
        end
        prevals(1:2) = prevals(2:3);
        prevals(3) = thisval;
    end
    % If the zero crossing was preceded by a lower minima,
    % then take/keep that as our starting point for the next step.
    if residuals(locs(l)) < curheight
        curheight = residuals(locs(l));
        maxnegdiff_ind = locs(l);
    end
end

if doplots
    figure(11); clf;
    hold on; plot( fourierSampling(maxnegdiff_ind), residuals(maxnegdiff_ind),'b*' );
    
end

% Trace back to where it is maximally different from our fit.
preval = residuals(maxnegdiff_ind-1)-residuals(maxnegdiff_ind);
if round(preval, 5)<0
    for i=maxnegdiff_ind-1:-1:2

        thisval = residuals(i-1)-residuals(i);

        if round(preval, 5)<=0 && round(thisval,5)<=0 % It should only be decreasing or flat- if it isn't anymore and heads upward, kick out.
            maxnegdiff_ind=i; 
        elseif thisval>0.03
            break;
        end
        preval = thisval;
    end
end

if doplots
    figure(thePlot); hold on;
    plot( fourierSampling(maxnegdiff_ind), fourierProfile(maxnegdiff_ind),'r*' );
    hold off;
    figure(11); plot(fourierSampling, residuals );
    plot( fourierSampling(maxnegdiff_ind), residuals(maxnegdiff_ind),'r*' );
    hold off;
end

minbound = 10;

maxamp = max(residuals(minbound:end))-min(residuals(minbound:end));


err = fourierProfile(1); %max(residuals(minbound:end).^2); 

shift = fourierSampling(maxnegdiff_ind+1);
shiftind = maxnegdiff_ind+1;
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
    x = [params.scale1 params.decay1 params.offset1 params.exp1 params.shift];
end


% fitParams = XToParams(x,params)
%
% Convert search params and base structure to filled in structure.
function params = XToParams(x,params)
params.scale1 = x(1);
params.decay1 = x(2);
params.offset1 = x(3);
params.exp1 = x(4);
% params.scale2 = x(4);
% params.decay2 = x(5);
params.shift = x(5);
end

% preds =  ComputeModelPreds(params,t)
%
% Compute the predictions of the model
function fullExp = ComputeModelPreds(params,freqBase)

fullExp = params.offset1 + params.scale1*params.exp1.^( -params.decay1 * (freqBase-params.shift) );

% bottomExpLoc = find(freqBase>params.shift);
% bottomExpTime = freqBase(bottomExpLoc);
% params.shift
% % The exponential must always line up with the other exponential function's
% % value!   
% maxmatch = fullExp(bottomExpTime(1))-params.scale2;
% 
% fullExp(bottomExpLoc) =  params.scale2*exp( -params.decay2 * (bottomExpTime-bottomExpTime(1)) );

end
