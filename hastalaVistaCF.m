%% hastalaVista CF modeling
%
% Short tutorial showing how to compute connective fields (Koen Haak, 2013) from fMRI time series
% in a transparent and straightforward way. Please feel free to modify,
% comment, extend and give feedback. 
% For details see: 
% http://www.spinozacentre.nl/dumoulin/PDFs/Haak-NI-2013.pdf
% https://www.frontiersin.org/articles/10.3389/fnins.2014.00339/full
% Nicolas Gravel, 12/2017
% University Medical Center Groningen
% Department of Experimental Ophthalmology
% NeuroImaging Center 
% E. n.g.gravel.araneda@rug.nl

close all
clear all
addpath(genpath('/Users/.../vistasoft-master'));
addpath(genpath('/Users/Nicolas/Dropbox (Personal)/githubReps/hastalaVista'));
mrVista 3
vw = VOLUME{1};
mrGlobals;
hemis = 'L';
roi_1 = 'V1'; roi_2 = 'V2'; roi_3 = 'V3';
ROIS{1} = strcat(hemis,roi_1);
ROIS{2} = strcat(hemis,roi_2);
ROIS{3} = strcat(hemis,roi_3);
vw = loadROI(vw, ROIS,[], [], 0, 1);
roiList = viewGet(vw,'roinames');
updateGlobal(vw);
scanID = cellstr(['VFM ';'RS1 ';'RS2 ']);

%% Retrieve ROI data
for roi = 1:numel(ROIS)
    volROI = vw.ROIs(roi);
    [~,roiInd{roi}] = intersectCols(vw.coords,volROI.coords);
    roiCoords{roi} = volROI.coords;
    distances{roi} = ccmVoxToVoxDist(volROI,vw,vw.mmPerVox);
end

%% Retrieve time series from each data type
global dataTYPES;
scan = 0;
for dataType = 4:6
    scan = scan + 1;
    vw = selectDataType(vw,dataType);
    curDataType = viewGet(vw,'curDataType');
    % Time series
    ts  = loadtSeries(vw, 1);  
    % Convert to % change
    dc   = ones(size(ts,1),1)*mean(ts);
    tSeries{scan} = ((ts./dc) - 1).*100;
end
    
%% Compute cortico-cortical RFs
TR = 1.5;
scan = 1;
sourceROI = 'V1';
targetROI = 'V3';
%%
% Defined ROI index and cortical distances
switch sourceROI
    case 'V1'
        idxSource = roiInd{1}; D = distances{1};
    case 'V2'
        idxSource = roiInd{2}; D = distances{2};
    case 'V3'
        idxSource = roiInd{3}; D = distances{3};
end
switch targetROI
    case 'V1'
        idxTarget = roiInd{1};
    case 'V2'
        idxTarget = roiInd{2};
    case 'V3'
        idxTarget = roiInd{3};
    case 'V2-V3'
        idxTarget = [idxV2 idxV3];
end

% Pick time series
ts = tSeries{scan}; 
tSeries_source = ts(:,idxSource); % Source ROI index
tSeries_target = ts(:,idxTarget); % Target ROI index
% Initialize CF model parameters: ROI coords
model.coords_source = vw.coords(:,idxSource);
model.coords_target = vw.coords(:,idxTarget);
% % Compute global signal 
% keep = find(~isnan(sum(ts)));
% m = size(ts(:,keep),2);
% % Regress global signal (optional)
% tSeries_global{scan} = (1/m)*ts(:,keep)*ones(m,1); 
 
% % Vestigial detrending - retrending code from the original implementation 
% % KH: Make trends to fit with the model (discrete cosine set)
% % The following parameters are needed for detrending with the function "rmMakeTrends"
% params = struct('stim',struct('nFrames',size(ts,1),'framePeriod',TR,'nDCT',3,'nUniqueRep',1)); % 3T
% [trends, ntrends, dcid]  = rmMakeTrends(params);
% trends = single(trends); % Create trends
% 
% % Remove trends source signals before generating the CF predictions 
% % no filtering before creating predictions!! (fixed)
% b = pinv(trends) * tSeries_source;
% tSeries_source = tSeries_source - trends*b;
% 
% % Remove trends from target signals
% b = pinv(trends) * tSeries_target;
% tSeries_target = tSeries_target - trends*b;

% Band pass filter
order = 4; fc = [0.01 0.1]; % Order and frequency band (Hz)
[b, a] = butter(order,2*TR*fc);
tSeries_source = filtfilt(b,a,double(tSeries_source));
tSeries_target = filtfilt(b,a,double(tSeries_target));

% Initialize CF model parameters: compute rss of target time series for variance computation later
model.rawrss = sum(tSeries_target.^2);

% Initialize CF models (# = Possible sizes * ROI size))
sigmas = linspace(0.0001,10,50); % Possible sizes
S = single(repmat(sigmas(:)',numel(1:size(D,1)),1)); % Sizes per source voxel
CF = single(zeros(numel(1:size(D,1)),numel(1:size(D,1))*length(sigmas)));
s = single(zeros(1,numel(1:size(D,1))*length(sigmas)));
candIdx = s; 
% Generate the CF models
for n = 1:size(D,1) 
    ii = (1:length(sigmas))+length(sigmas)*(n-1); % Number of candidate models depends on possible sizes
    candIdx(ii) = single(repmat(idxSource(n),1,numel(sigmas))); % Source indices associated with predictions
    X = single(repmat(D(:,n),1,numel(sigmas))); % Cortical distances
    s(ii) = single(sigmas); % Candidate CF sizes
    CF(:,ii) = single(exp(-1.*((X.^2)./(2.*S.^2)))); % Generate candidate CFs
end
predictions = tSeries_source(:,1:size(D,1))*CF; % Generate predictions (convolve CF with source signals)
% Initialiazie CF model parameters
% model.b = zeros(1,size(tSeries_target,2),ntrends); % Initialize betas 
model.b = zeros(1,size(tSeries_target,2)); % Initialize betas
model.mse = ones(1,size(tSeries_target,2)).*Inf; % Initialize mean squared error
model.sigmas = zeros(1,size(tSeries_target,2)) + 0.001; % Initialize sigmas
model.sourceROIidx = idxSource'; % Global source voxel index (in mrVista volume)
model.targetROIidx = idxTarget'; % Global target voxel index (in mrVista volume)

% Search the best fit between the CF-derived signals and the target ROI signals
tic
for n = 1 : length(predictions) % Iterate trhough candidate models 
    % Minimum MSE fit
    % X = [predictions(:,n) trends]; % Original implementation 
    X = predictions(:,n); 
    [b,stdx,mse] = lscov(X,single(tSeries_target)); % Least squares linear regression
    nkeep = b(1,:) < 0; mse(nkeep) = Inf; % Keep positive betas
    mseMin = mse < model.mse; % Minimize mse
    % Update model parameters
    model.mse(mseMin) = mse(mseMin); % CF mse
    % model.b([1 dcid+1],mseMin) = b(:,mseMin); % Standard error (vestigial)
    model.b(:,mseMin) = b(:,mseMin); % Standard error 
    model.sigmas(mseMin) = s(n); % CF size
    model.CFidx(mseMin) = candIdx(n); % Global CF index
end;
toc

% KH: Correct lscov. It returns the mean rss. To maintain compatibility with
% the sum rss this function expects, we have to multiply by the divisor.
% model.rss = single(model.mse.*(size(predictions,1)-size(trends,2)+1)); 
model.rss = single(model.mse.*(size(predictions,1)-1+1));
% NG: Note that if denoising steps are added, additional corrections should be
% applied.
% Variance explained: varexp = 1 - (rss ./ rawrss);
model.ve = 1 - (model.rss ./ model.rawrss); 
model.ve(~isfinite(model.ve)) = 0;
model.CFcoords = vw.coords(:,model.CFidx);

%% Associate pRF-derived retinotopic maps to CF maps
pRFmodel = ['./Gray/AveragesVFM/' 'retModel-20161013-181240-sFit.mat'];
%%
% Load pRF model
rm = load(pRFmodel);
x0_sourceROI = []; y0_sourceROI = [];
for n = 1:size(model.CFidx,2)
    x0_sourceROI = [x0_sourceROI rm.model{1}.x0(model.CFidx(n))];
    y0_sourceROI = [y0_sourceROI rm.model{1}.y0(model.CFidx(n))];
end
% convert to polar coordinates
[pol, ecc] = cart2pol(x0_sourceROI, y0_sourceROI);
pol = mod(pol, 2*pi);
% KH: must do some sanity checks here
pol(pol == Inf)  = max(pol(isfinite(pol(:))));
pol(pol == -Inf) = min(pol(isfinite(pol(:))));
pol(isnan(pol)) = 0;
pol = max(pol,0);
pol = min(pol,2*pi);
ecc(ecc == Inf)  = max(ecc(isfinite(ecc(:))));
ecc(ecc == -Inf) = min(ecc(isfinite(ecc(:))));
ecc(isnan(ecc)) = 0;
% Now retrieve pRF maps derived maps
model.ecc =  ecc;
model.pol = pol;
save CFmodel model


%% Plot CF maps
close all
clear all
addpath(genpath('/Volumes/Data 1/hastalaVistaCFmodeling/hastalaVista'));
mrVista 3
vw = VOLUME{1};
mrGlobals;
load  CFmodel
hemis = 'L';

%% Load mrMesh (open 3D window)
switch hemis
    case 'L'
        meshName = 'lmesh_inflated.mat';
    case 'R'
        meshName = 'rmesh_inflated.mat';
end
mesh = strcat('./Anatomy/', meshName);
vw = meshLoad(vw, mesh, 1);
% Load mesh setting
msh = viewGet(vw, 'selectedMesh');
switch hemis
    case 'L'
        meshRetrieveSettings(msh, 'lMesh');
    case 'R'
        meshRetrieveSettings(msh, 'rMesh');
end
updateGlobal(vw);
vw = getCurView;  %% mod for show on mesh plot to work
map = cell(1,viewGet(vw,'numscans'));
map{viewGet(vw,'curscan')} = zeros(1,size(vw.coords,2));
map{viewGet(vw,'curscan')}(:) = 0;
updateGlobal(vw);


%% Plot to 3D mesh
% Choose overlay
% Fix for left hemisphere
% for n = 1:length( model.pol)
%     if  model.pol(n) > pi
%          model.pol(n) = 2*pi -  model.pol(n); % correct left hemisphere
%     end
% end
% overlay = model.ve; colors = 'jetCmap';
overlay = model.pol; colors = 'hsvCmap';
roiInd = model.targetROIidx;
%%
map{viewGet(vw,'curscan')}(roiInd) = overlay;
vw = viewSet(vw, 'map', map);
map{viewGet(vw,'curscan')}(:) = -1;
map{viewGet(vw,'curscan')}(roiInd) = 1;
vw = viewSet(vw, 'co', map);
vw.ui.mapMode = setColormap(vw.ui.mapMode, colors);
vw = setDisplayMode(vw, 'map');
vw = refreshScreen(vw);
vw = setClipMode(vw, 'map', [min(overlay) max(overlay)]);
% mrmPreferences can be modified from its default parameters to avoid
% smoothing and set a higher transparency for the data overlay
vw.ui.showROIs = -1; vw = refreshScreen(vw,3);
% meshColorOverlay(vw,showData,dataOverlayScale,dataThreshold)
vw = meshColorOverlay(vw,[],[],0);
updateGlobal(vw);
