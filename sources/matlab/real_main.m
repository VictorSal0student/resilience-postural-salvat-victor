%% ==========================================================
%                   LOCOMOTOR RESILIENCE ANALYSIS
% ==========================================================
% Goal:
%   - Measure center of mass (COM) and resynchronization recoveries after auditory perturbations
%     in near-ecological conditions.
%   - Compare responses between young and older adults.
%
% Input:
%   - Sacrum vertical oscillations stored in a .mat file:
%       DATA.Sacrum.filtered_value
%   - Left and right heel vertical oscillations
%       DATA.Sacrum.filtered_value
%
% Output:
%   - Reference trajectory in reconstructed state space (RefTrajSmooth)
%   - Recovery times (recovery_Times) and indices (recovery_indices)
%   - Persistence differences post-perturbation
%   - Phase-space reconstructions for each perturbation
%   - CSV summary files for group-level statistical analysis
%
% Notes:
%   - Sampling frequency (fs) is assumed to be 100 Hz
%   - Two types of perturbations: Fast & Slow
%   - Stability thresholds defined by ellipses at 1σ, 2σ, 3σ
% ==========================================================

%% --- Cleaning ---

clear
close all
clc

%% --- Constants & Parameters

fs = 100;   % Sampling frequency [Hz]
threshold_detrend = 0.01;  % mm, threshold for drift correction
maxLag = 100;   % Max lag for AMI
numBins = 20;   % Histogram bins for AMI
maxEmbeddingDimension = 10;
FNN_threshold = 0.1;

% Define groups
young_IDs = [1, 2, 3, 4, 5, 7, 8, 9, 12, 17, 18, 19, 20, 21, 22, 23, 24, 27, 36, 37];           % Group : Young   
elderly_IDs = [6, 10, 11, 13, 14, 15, 16, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40];  % Group : Aging
slow_first = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 34, 35, 37, 38, 39, 40];           % First perturbation is slow

                                    %% --- Open the file containing data ---

addpath '...\DATA\IMF'; % Insert your file path
fileName = '040_Cued_imf.mat'; % 17 & 40
DATA = load(fileName).DATA;

% Extract COM coordinates
mean_z = DATA.Sacrum.downsampled_value(:,3);

% Time vector
t = (1:length(mean_z))';  % Time in seconds

                                    %% --- Optional : Polynomial regression ---

% Fit a 1st-order polynomial to detect slow drift
p = polyfit(t, mean_z, 1);
baseline = polyval(p, t);
amplitude_drift = max(baseline) - min(baseline); % Slope

if amplitude_drift > threshold_detrend
    DATA.Detrended.downsampled_value = mean_z - baseline;
    applied_correction = true;
    fprintf('Applied drift correction (drift amplitude: %.2f mm)\n', amplitude_drift);
else
    DATA.Detrended.downsampled_value = mean_z;
    applied_correction = false;
    fprintf('No drift correction applied (drift amplitude: %.2f mm)\n', amplitude_drift);
end

%% Low pass filter

forder = 4;
cutfreq = 5;

% Create the filter coefficients
[b, a] = butter(forder, cutfreq / fs);

fieldNames = fieldnames(DATA);  % Get the field names of the struct

% Iterate over each field
for i = 1:length(fieldNames)
    fieldName = fieldNames{i};

    % Check if the field contains a 'value' field (to avoid errors)
    if isfield(DATA.(fieldName), 'downsampled_value')
        data = DATA.(fieldName).downsampled_value;

        % Remove or replace NaN/Inf values by interpolation or setting to zero
        if isnumeric(data)
            % Cleaning : replace INf and NaN
            data(~isfinite(data)) = NaN; % Take all Inf/-Inf as NaN
            data = fillmissing(data, 'linear', 1); % Interpolation on lines
            data = fillmissing(data, 'linear', 2); % Interpolation on rows
            % If NaN persists
            data(isnan(data)) = 0;
        end

        % Apply the filter to the cleaned data
        filtered_data = filtfilt(b, a, data);

        % Store the filtered data back into the struct
        DATA.(fieldName).filtered_value = filtered_data;
    end
end

% Extract COM coordinates
mean_x = DATA.Sacrum.filtered_value(:,1);
mean_y = DATA.Sacrum.filtered_value(:,2);

                                    %% --- Time delay embedding method ---

% --- This part aims to define the parameters (tau & dim) for state space reconstruction (see Wurdeman, 2016) ---

%% ----------  1. Average Mutual Information ----------

addpath '...\functions'; % Insert your file path

COM = DATA.Detrended.filtered_value;
amiValues = zeros(maxLag, 1);

for lag = 1:maxLag
    shiftedData  = COM(1:end-lag);
    originalData = COM(lag+1:end);

    edges = linspace(min(COM), max(COM), numBins+1);

    [jointCounts, ~, ~] = histcounts2(originalData, shiftedData, edges, edges);
    jointProb = jointCounts / sum(jointCounts(:));

    px = sum(jointProb, 2); % Marginal probabilities (original)
    py = sum(jointProb, 1); % Marginal probabilities (shifted)

    mi = 0;
    for i = 1:numBins
        for j = 1:numBins
            if jointProb(i, j) > 0
                mi = mi + jointProb(i, j) * log(jointProb(i, j) / (px(i) * py(j)));
            end
        end
    end
    amiValues(lag) = mi;
end

[~, Tau] = findpeaks(-amiValues, 'NPeaks', 1);
fprintf('Optimal time delay (Tau): %d\n', Tau);

% Optional visualization
figure; hold on;
plot(1:maxLag, amiValues, '-o');
plot(Tau, amiValues(Tau), 'rx', 'LineWidth', 2);
xlabel('Time Delay (tau)');
ylabel('Average Mutual Information (AMI)');
title('Average Mutual Information for Time Delay Selection');
grid on;

%% ----------  2. False Neirest Neighbors ----------

fnnRates = zeros(1, maxEmbeddingDimension);

for dim = 1:maxEmbeddingDimension
    fnnRates(dim) = FNN_function(COM, dim, Tau);
end

Dim = find(fnnRates < FNN_threshold, 1, 'first');
fprintf('Optimal embedding dimension (Dim): %d\n', Dim);

% Optional visualization
figure; hold on;
plot(1:maxEmbeddingDimension, fnnRates, '-s');
plot(Dim, fnnRates(Dim), 'rx', 'LineWidth', 2);
title('False Nearest Neighbors vs. Embedding Dimension');
xlabel('Embedding Dimension');
ylabel('FNN Rate (%)');
grid on;

                                    %% --- State space reconstruction ---

XR = phaseSpaceReconstruction(COM, Tau, Dim);

% Keep only 3 dimensions
Original = XR(:,1);
Delayed1 = XR(:,2);
Delayed2 = XR(:,3);

%% --- Beeps detection & disturbances ---

% Exctraction and preparation of the analog signal (beeps)
Analog = DATA.Analog; % Auditory cueing
Analog.value = abs(Analog.value); % Absolute values to avoid negatives
fs_analog = Analog.Rate; % Sampling frequency [Hz]

% Beeps detection in the signal
[tempo_value, locs_tempo] = findpeaks(Analog.value, ...
    'MinPeakHeight', 35, ...        % Minimal amplitude
    'MinPeakDistance', 100);        % Minimum distance between peaks (samples)
bip_times = locs_tempo / fs_analog; % Conversion to seconds

% Calculation of intervals between steps
bip_intervals = diff(bip_times);
baseline_interval = mean(bip_intervals(1:10)); % Average of the first 10 = baseline

% Tolerance threshold around baseline (±15%)
tolerance = 0.15;
lower_bound = baseline_interval * (1 - tolerance);
upper_bound = baseline_interval * (1 + tolerance);

% --- Paramètres ---
n_seq = 3;   % nombre minimum d'intervalles consécutifs requis

% --- Rapid disturbance detection (shortest interval) ---
idx_fast = NaN;
for i = 1:(length(bip_intervals)-n_seq+1)
    seq = bip_intervals(i:i+n_seq-1);
    if all(seq < lower_bound)   % au-dessus du seuil rapide
        idx_fast = i;           % premier indice de la séquence
        break;
    end
end

% --- Slow disturbance detection (longest interval) ---
idx_slow = NaN;
for i = 1:(length(bip_intervals)-n_seq+1)
    seq = bip_intervals(i:i+n_seq-1);
    if all(seq > upper_bound)   % au-dessus du seuil lent
        idx_slow = i;           % premier indice de la séquence
        break;
    end
end

% --- Rapid disturbance onset search ---
local_range_fast = max(1, idx_fast-5) : idx_fast;
is_anomalous_fast = bip_intervals(local_range_fast) < lower_bound | ...
                    bip_intervals(local_range_fast) > upper_bound;
first_anomaly_fast = find(is_anomalous_fast, 1, 'first');
perturbation_fast_idx = local_range_fast(first_anomaly_fast);

% --- Slow disturbance onset search ---
local_range_slow = max(1, idx_slow-5) : idx_slow;
is_anomalous_slow = bip_intervals(local_range_slow) < lower_bound | ...
                    bip_intervals(local_range_slow) > upper_bound;
first_anomaly_slow = find(is_anomalous_slow, 1, 'first');
perturbation_slow_idx = local_range_slow(first_anomaly_slow);

% Conversion in time
pert_idx_raw = [perturbation_fast_idx, perturbation_slow_idx];
pert_times_raw = bip_times(pert_idx_raw); % +1 because diff shifts

% Chronological sorting of disruptions
[perturbation_times, sort_idx] = sort(pert_times_raw);
perturbation_indices = pert_idx_raw(sort_idx);

% Console display
fprintf('Fast disturbance onset : %.3f s\n', perturbation_times(1));
fprintf('Slow disturbance onset  : %.3f s\n', perturbation_times(2));

% --- Graphical verification ---
figure;
plot(bip_times(2:end), bip_intervals, '-o'); hold on
yline(baseline_interval, '--k', 'Baseline');
yline(lower_bound, ':r', 'Thresholds');
yline(upper_bound, ':r');
plot(bip_times(perturbation_fast_idx+1), bip_intervals(perturbation_fast_idx), ...
    'ro', 'MarkerSize', 10, 'DisplayName', 'Fast pert start');
plot(bip_times(perturbation_slow_idx+1), bip_intervals(perturbation_slow_idx), ...
    'bo', 'MarkerSize', 10, 'DisplayName', 'Slow pert start');
legend('Location', 'best');
xlabel('Time (s)');
ylabel('Inter-bip interval (s)');
title('Detection of Perturbations');
grid on

%% --- Creation of reference trajectory ---

PreIndices   = 1:round(perturbation_times(1)) * 100 - Tau;
PreOriginal  = XR(PreIndices, 1);
PreDelayed1  = XR(PreIndices, 2);
PreDelayed2  = XR(PreIndices, 3);

% Centroid
Xo = mean(PreOriginal, 'omitnan');
Yo = mean(PreDelayed1, 'omitnan');
Zo = mean(PreDelayed2, 'omitnan');

% Plane fitting
fitresult = fct_createFit1(PreOriginal, PreDelayed1, PreDelayed2);
normalVec = [-fitresult.p10, -fitresult.p01, 1];

% Rotation to align plane with XY
rotationVector  = vrrotvec([0 0 1], normalVec);
rotationMatrix  = vrrotvec2mat(rotationVector);
ProjPre         = ([PreOriginal - Xo, PreDelayed1 - Yo, PreDelayed2 - Zo] * rotationMatrix);
Projxp1         = ProjPre(:,1);
Projyp1         = ProjPre(:,2);

% Phase angle
phaseAnglePre = atan2d(Projyp1, Projxp1) + 180;
[phaseSorted, Idx] = sort(round(phaseAnglePre));

% Sort points
PreOriginalS = PreOriginal(Idx);
PreDelayed1S = PreDelayed1(Idx);
PreDelayed2S = PreDelayed2(Idx);

%% --- Reference trajectory via periodic Fourier fit (Ravi-like) ---

% 1) Prepare the phase to comply with the periodicity (duplicate the
% period)
phase_deg   = phaseSorted(:);                          % [0..360]
x1          = PreOriginalS(:);
y1          = PreDelayed1S(:);
z1          = PreDelayed2S(:);

% Retire NaNs
valid       = isfinite(phase_deg) & isfinite(x1) & isfinite(y1) & isfinite(z1);
phase_deg   = phase_deg(valid);
x1          = x1(valid);
y1          = y1(valid);
z1          = z1(valid);

% Duplicate a second period to neatly 'close' the Fourier (360° further on)
phase_fit   = [phase_deg; phase_deg + 360];
x_fit       = [x1;        x1];
y_fit       = [y1;        y1];
z_fit       = [z1;        z1];

% 2) Choice of Fourier order (generally 8)

fourier_order = 8;

use_curvefitting_toolbox = exist('fit','file')==2;

phaseAngleRef = (0:359)';   % Ouput phase grid
if use_curvefitting_toolbox
    % ----- Option A : Curve Fitting Toolbox -----
    ft = fittype(sprintf('fourier%d', fourier_order));
    opts = fitoptions(ft);
    % (Optionnal) Adjust options if necessary :
    % opts.Robust = 'Bisquare'; % Usefull if outliers
    % opts.Normalize = 'on';    % Usefull to normalize

    fit_x = fit(phase_fit, x_fit, ft, opts);
    fit_y = fit(phase_fit, y_fit, ft, opts);
    fit_z = fit(phase_fit, z_fit, ft, opts);

    Newxp = feval(fit_x, phaseAngleRef);
    Newyp = feval(fit_y, phaseAngleRef);
    Newzp = feval(fit_z, phaseAngleRef);

else
    % Regression matrix A for training points
    phi_fit = phase_fit(:);   % column Mx1
    S_fit   = sin( phi_fit * (w*(1:N)) );   % MxN
    C_fit   = cos( phi_fit * (w*(1:N)) );   % MxN
    A       = [ones(numel(phi_fit),1), S_fit, C_fit];  % M x (1+2N)

    % Minimum squares resolution
    bx = A \ x_fit;
    by = A \ y_fit;
    bz = A \ z_fit;

    % Prediction matrix B for grid 0.359
    phi_ref = phaseAngleRef(:);       % 360x1
    S_ref   = sin( phi_ref * (w*(1:N)) );   % 360xN
    C_ref   = cos( phi_ref * (w*(1:N)) );   % 360xN
    B       = [ones(numel(phi_ref),1), S_ref, C_ref];

    Newxp = B * bx;
    Newyp = B * by;
    Newzp = B * bz;
end

% 3) Periodic reference trajectory (360 x 3)
RefTrajSmooth = [Newxp, Newyp, Newzp];

%% --- Local standard deviation & Ellipse creation ---

resolution = 360;

% Give standard deviation for every angle
[stdx, stdy, stdz] = fct_std_RefTraj( ...
    PreOriginalS, ...
    phaseAngleRef, ...
    [PreOriginalS, PreDelayed1S, PreDelayed2S, phaseSorted], ...
    RefTrajSmooth(:,1), ...
    RefTrajSmooth(:,2), ...
    RefTrajSmooth(:,3), ...
    resolution);

% Option : smooth global standards
stdx_smooth = smoothdata(stdx, 'gaussian', 30);
stdy_smooth = smoothdata(stdy, 'gaussian', 30);
stdz_smooth = smoothdata(stdz, 'gaussian', 30);

% 2) Local standard deviations in the normal/binormal reference frame
PreDataPhaseAngleRS = [PreOriginalS, PreDelayed1S, PreDelayed2S, phaseSorted];
[stdLocalN, stdLocalB, ~] = computeLocalStd(RefTrajSmooth, PreDataPhaseAngleRS, phaseAngleRef);

% Option : smoothing
stdLocalN_smooth = smoothdata(stdLocalN, 'gaussian', 30);
stdLocalB_smooth = smoothdata(stdLocalB, 'gaussian', 30);

% 3) Construction of local ellipses
resolution3 = 50; % ellipse points
[T, N, B] = computeLocalFrames(RefTrajSmooth);
ellipse3d = fct_ellipseOrientation_local(RefTrajSmooth, stdLocalN_smooth, stdLocalB_smooth, N, B, resolution3);

% % 4) Display : some local ellipses
% nbEllipsesToPlot = 1;
% startIndex = round(0.3 * length(phaseAngleRef));
% indices = startIndex : startIndex + nbEllipsesToPlot - 1;
% 
% figure; hold on; grid on; axis equal;
% xlabel('t'); ylabel('t+tau'); zlabel('t+2tau');
% title('Local ellipses on the trajectory');
% plot3(RefTrajSmooth(:,1), RefTrajSmooth(:,2), RefTrajSmooth(:,3), ...
%       'LineWidth', 2, 'Color', [0.6 0.2 0.8]);
% 
% for idx = indices
%     j = (idx - 1) * 3 + 1;
%     plot3(ellipse3d(:, j), ellipse3d(:, j+1), ellipse3d(:, j+2), ...
%           'Color', [237 179 20]/255, 'LineWidth', 1.2);
% end
% view(3); camlight; lighting gouraud;
% xlim([-25 25])
% ylim([-25 25])
% zlim([-25 25])

%% --- Torus Ellipses Creation (1σ, 2σ, 3σ) ---

% 1) Ellipse axes for each angle
maxDeviation = NaN(length(phaseAngleRef), 6);
for i = 1:length(phaseAngleRef)
    stds = [stdx(i), stdy(i), stdz(i)];
    sortedStds = sort(stds, 'descend');
    maxDeviation(i,4:6) = sortedStds; % [large, medium, small]
end

% 2) Multiple scales
maxDev1 = maxDeviation;
maxDev2 = maxDeviation;  maxDev2(:,4:6) = 2 * maxDev2(:,4:6);
maxDev3 = maxDeviation;  maxDev3(:,4:6) = 3 * maxDev3(:,4:6);

% 3) Ellipses for each σ
ellipses1 = fct_ellipseOrientation_local(RefTrajSmooth, maxDev1(:,4), maxDev1(:,5), N, B, resolution3);
ellipses2 = fct_ellipseOrientation_local(RefTrajSmooth, maxDev2(:,4), maxDev2(:,5), N, B, resolution3);
ellipses3 = fct_ellipseOrientation_local(RefTrajSmooth, maxDev3(:,4), maxDev3(:,5), N, B, resolution3);

% Close if trajectory is closed
ellipsesList = {ellipses1, ellipses2, ellipses3};
for k = 1:numel(ellipsesList)
    ellipsesList{k}(:,end+1:end+3) = ellipsesList{k}(:,1:3);
end
[ellipses1, ellipses2, ellipses3] = deal(ellipsesList{:});

%% --- 4) Clean visualization : One ellipse (1σ, 2σ, 3σ) ---
nbEllipsesToPlot = 1;
startIndex = round(0.3 * length(phaseAngleRef));
indices = startIndex : startIndex + nbEllipsesToPlot - 1;

figure('Color', 'w'); hold on; grid on; axis equal tight;

% --- Colorization ---
colorRef = [0.5 0.1 0.8];        % Purple : reference trajectory
col1 = [229/255 78/255 209/255]; % Pink (1σ)
col2 = [237 179 20]/255;         % Yellow (2σ)
col3 = [191/255 216/255 52/255]; % Green (3σ)

% --- Title & labels ---
title('State space reconstruction', 'FontSize', 14, 'FontWeight', 'normal');
xlabel('X(t)', 'FontSize', 20, 'FontWeight', 'normal');
ylabel('X(t + \tau)', 'FontSize', 20, 'FontWeight', 'normal');
zlabel('X(t + 2\tau)', 'FontSize', 20, 'FontWeight', 'normal');

% --- Principal trajectory ---
plot3(RefTrajSmooth(:,1), RefTrajSmooth(:,2), RefTrajSmooth(:,3), ...
    'LineWidth', 2, 'Color', colorRef, 'DisplayName', 'Reference trajectory');

% --- Local ellipses (1σ, 2σ, 3σ) ---
for idx = indices
    j = (idx - 1) * 3 + 1;

    % 1σ
    plot3(ellipses1(:, j), ellipses1(:, j+1), ellipses1(:, j+2), ...
        'Color', col1, 'LineWidth', 1.4, 'DisplayName', 'Local ellipse (1σ)');
    % 2σ
    plot3(ellipses2(:, j), ellipses2(:, j+1), ellipses2(:, j+2), ...
        'Color', col2, 'LineWidth', 1.4, 'DisplayName', 'Local ellipse (2σ)');
    % 3σ
    plot3(ellipses3(:, j), ellipses3(:, j+1), ellipses3(:, j+2), ...
        'Color', col3, 'LineWidth', 1.4, 'DisplayName', 'Local ellipse (3σ)');
end

% --- Parameters for plotting ---
legend('Location', 'northeastoutside');
view(35, 25);
camlight headlight; lighting gouraud; material dull;
set(gca, 'FontSize', 15, 'LineWidth', 1, 'Box', 'on', ...
         'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);

%% --- 5) Clean visualization : Ellipse torus ---

figure('Color', 'w'); hold on; grid on;

% --- Couleurs ---
colorRef = [0.5 0.1 0.8];         % Purple : reference trajectory
colorPoint = [0.9 0.1 0.1];       % Red : Centroid
colorEllipse = [237 179 20]/255;  % Yellow : ellipse torus (1σ)
colorTraj = [0 0 0];              % Black : Trajectory of sacrum (z-axis)

% --- Reference trajectory ---
hRef = plot3(RefTrajSmooth(:,1), RefTrajSmooth(:,2), RefTrajSmooth(:,3), ...
    'Color', colorRef, 'LineWidth', 2.5, 'DisplayName', 'Reference trajectory');

% --- Centroid ---
hPoint = scatter3(Xo, Yo, Zo, 60, colorPoint, 'filled', 'DisplayName', 'Centroid');

% --- Ellipse torus ---
resPlusOne = resolution + 1;
for j = 1:3:resPlusOne*3
    if j == 1
        hEllipse = plot3(ellipses2(:,j), ellipses2(:,j+1), ellipses2(:,j+2), ...
            'Color', colorEllipse, 'LineWidth', 0.8, 'DisplayName', 'Ellipse torus');
    else
        plot3(ellipses2(:,j), ellipses2(:,j+1), ellipses2(:,j+2), ...
            'Color', colorEllipse, 'LineWidth', 0.8, 'HandleVisibility', 'off');
    end
end

% --- Perturbated trajectory ---
hPert = plot3(Original(15000:15500), Delayed1(15000:15500), Delayed2(15000:15500), ...
    'Color', colorTraj, 'LineWidth', 1.5, 'DisplayName', 'Sacrum trajectory (z-axis)');

% --- Title & labels ---
title('State space reconstruction', 'FontSize', 20, 'FontWeight', 'normal');
xlabel('X(t)', 'FontSize', 15, 'FontWeight', 'normal');
ylabel('X(t + \tau)', 'FontSize', 15, 'FontWeight', 'normal');
zlabel('X(t + 2\tau)', 'FontSize', 15, 'FontWeight', 'normal');

% --- Parameters for plotting ---
legend([hRef, hPoint, hEllipse, hPert], 'Location', 'northeastoutside');
axis equal
axis tight       % <-- Adjust the limits
view(35, 25);
camlight headlight; lighting gouraud; material dull;
set(gca, 'FontSize', 15, 'LineWidth', 1, 'Box', 'on', ...
         'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'ZColor', [0.2 0.2 0.2]);

                                %% --- Resynchronization preparation ---

% This section prepares the probe and target timestamps to compute relative phase

% --- 1) Steps ---
[~, Peak] = findpeaks(-COM); % Negative peaks of sacrum

% --- 2) Timestamps ---
timestamps_target = locs_tempo / 5; % Convert tempo indices to seconds
timestamps_target = timestamps_target(:);

% Combine left and right heel strike indices
index_timestamps_probe = Peak; % Peak or sort([poses_G_idx; poses_D_idx])
timestamps_probe = index_timestamps_probe(1:end-1);

% --- 3) Match probe to closest target ---
max_thresh = 60; % Max threshold (seconds) for matching

% Preallocate arrays
timestamps_probe_valid = NaN(size(timestamps_probe));
timestamps_target_matched = NaN(size(timestamps_probe));
count = 0;

for i = 1:length(timestamps_probe)
    t_probe = timestamps_probe(i);

    % Find the closest target (bip)
    [delta, idx_closest] = min(abs(timestamps_target - t_probe));

    if delta <= max_thresh
        count = count + 1;
        timestamps_probe_valid(count) = t_probe;
        timestamps_target_matched(count) = timestamps_target(idx_closest);
    end
end

% Remove unused NaN entries
timestamps_probe_valid      = timestamps_probe_valid(1:count);
timestamps_target_matched   = timestamps_target_matched(1:count);

% --- 4) Compute relative phase ---
phivec = relative_phase_from_discrete_events(t, timestamps_probe_valid, timestamps_target_matched); % returns radians

% Convert to degrees and center around [-180, 180] for visualization
phivec_deg = rad2deg(phivec);
phivec_deg_centered = mod(phivec_deg + 180, 360) - 180;

% Optional filtering: remove extreme outliers (> ±120°)
phivec_deg_filtered = phivec_deg_centered(abs(phivec_deg_centered) < 120);

% --- 5) Time vector for plotting ---
timestamps_probe = timestamps_probe_valid(1:length(phivec_deg_centered));
timestamps_probe = timestamps_probe / 100; % scale to seconds if needed

%% --- RECOVERY ANALYSIS - Classic version (Bank et al., 2011) ---

timestamps_probe_valid = timestamps_probe_valid / 100;
timestamps_target_matched = timestamps_target_matched / 100;

% --- 1) Step times ---
step_times = diff(timestamps_probe_valid);

% --- 2) IBI's local reference ---
IBI_ref_local = NaN(size(step_times));
for i = 1:length(step_times)
    if i+1 <= length(timestamps_target_matched)
        IBI_ref_local(i) = timestamps_target_matched(i+1) - timestamps_target_matched(i);
    end
end
IBI_ref_local = fillmissing(IBI_ref_local,'linear','EndValues','nearest');

% --- 3) Asynchrony ---
asynchrony = step_times - IBI_ref_local;

% --- 4) Analysis by perturbation ---
resync_time_steps_all = NaN(1,length(perturbation_indices));
resync_time_sec_all   = NaN(1,length(perturbation_indices));

figure;
colors = lines(length(perturbation_indices));

for p = 1:length(perturbation_indices)
    pert_idx = min(perturbation_indices(p), length(asynchrony)-1);

    % --- Baseline : 10 steps before perturbation ---
    start_idx = max(1, pert_idx-10);
    baseline = asynchrony(start_idx:pert_idx-1);

    mean_pre = mean(baseline);
    std_pre  = std(baseline);
    lower_lim = mean_pre - 2*std_pre;
    upper_lim = mean_pre + 2*std_pre;

    % --- Post-perturbation peak (maximal asynchrony) ---
    post_range = pert_idx:min(pert_idx+6, length(asynchrony));
    [~, peak_rel] = max(abs(asynchrony(post_range) - mean_pre));
    peak_idx = post_range(1) + peak_rel - 1;

    % --- Resynchonization detection ---
    N = length(asynchrony);
    recovered = false;

    fprintf('\n--- Perturbation %d ---\n', p);
    fprintf('Baseline mean=%.3f | std=%.3f | 2σ=[%.3f %.3f]\n', mean_pre, std_pre, lower_lim, upper_lim);
    fprintf('Testing windows from step %d (peak)\n', peak_idx);

    for i = peak_idx:N-2
        % Resynchronization: 8 consecutive windows of 3 steps in stability zone
        stable = true;
        for j = 0:7
            idx_start = i + j;
            idx_end   = idx_start + 2;
            if idx_end > N
                stable = false; 
                break;
            end
            win_vals = asynchrony(idx_start:idx_end);
            win_mean = mean(win_vals);

            if win_mean < lower_lim || win_mean > upper_lim
                stable = false;
                break;
            end
        end

        if stable
            resync_idx = i+1; % First step stable
            recovered = true;
            fprintf('✅ Stability [%d-%d] windows on 8 consecutive detrended windows \n', i, i+2);
            break;
        end
    end

    % --- Resynchronizatio time / steps ---
    if recovered && resync_idx > peak_idx
        resync_time_steps = resync_idx - peak_idx;
        i1 = min(resync_idx+1, length(timestamps_probe_valid));
        i0 = min(peak_idx+1, length(timestamps_probe_valid));
        resync_time_sec = timestamps_probe_valid(i1) - timestamps_probe_valid(i0);
    else
        resync_time_steps = NaN;
        resync_time_sec   = NaN;
    end

    resync_time_steps_all(p) = resync_time_steps;
    resync_time_sec_all(p)   = resync_time_sec;

    fprintf('Perturbation %d -> Recovery: %.2f s (%d steps)\n', p, resync_time_sec, resync_time_steps);

    % --- Parameters for plotting ---
    subplot(length(perturbation_indices),1,p); hold on;

    % Stability zone
    fill([1 length(asynchrony) length(asynchrony) 1], ...
         [lower_lim lower_lim upper_lim upper_lim], ...
         [0.9 0.9 0.9], ...
         'FaceAlpha', 0.3, ...
         'EdgeColor', 'none', ...
         'DisplayName', 'Stability zone');

    % Asynchrony
    plot(asynchrony, '-o', ...
        'Color', colors(p,:), ...
        'MarkerFaceColor', colors(p,:), ...
        'MarkerEdgeColor', 'none', ...
        'LineWidth', 1.8, ...
        'MarkerSize', 5, ...
        'DisplayName', 'Asynchrony');

    % Perturbation
    xline(pert_idx, '--', ...
        'Color', colors(p,:), ...
        'LineWidth', 2, ...
        'DisplayName', 'Perturbation');

    % Peak
    plot(peak_idx, asynchrony(peak_idx), 'v', ...
        'Color', colors(p,:), ...
        'MarkerFaceColor', colors(p,:), ...
        'MarkerSize', 10, ...
        'DisplayName', 'Peak');

    % Recovery
    if recovered && resync_idx > peak_idx
        plot(resync_idx, asynchrony(resync_idx), 's', ...
            'Color', [0 0.6 0], ...          % vert foncé
            'MarkerFaceColor', [0.3 1 0.3], ...
            'MarkerSize', 10, ...
            'LineWidth', 1.5, ...
            'DisplayName', 'Recovery');
    end

    % Stability zone
    yline(mean_pre, '--k', 'LineWidth', 1.2, 'DisplayName', 'Baseline mean');
    yline(lower_lim, ':k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    yline(upper_lim, ':k', 'LineWidth', 1.2, 'HandleVisibility', 'off');

    % Titles & labels
    xlabel('Step index', 'FontWeight', 'bold');
    ylabel('Asynchrony (s)', 'FontWeight', 'bold');
    title(sprintf('Perturbation %d - Recovery: %.2f s (%d steps)', ...
          p, resync_time_sec, resync_time_steps), ...
          'FontSize', 11, 'FontWeight', 'bold');

    legend('Location', 'best', 'Box', 'off');
    grid on; box off;
    set(gca, 'FontSize', 10, 'LineWidth', 1.1);
end

disp(round(length(timestamps_probe_valid) / 4.005));
fprintf('%.3f\n', mean(step_times));
fprintf('%.3f\n', mean(asynchrony(1:perturbation_indices(1)-1) * 1000));
fprintf('%.3f\n', mean(asynchrony(peak_idx+resync_time_steps_all(1)+1:perturbation_indices(2)-1) * 1000));
fprintf('%.3f\n', mean(asynchrony(peak_idx+resync_time_steps_all(2)+1:end) * 1000));

resync_times = [resync_time_sec_all(1) resync_time_sec_all(2)];

                                    %% --- COM recovery after perturbations ---

%% I. Euclidean distances & classification with T1σ, T2σ, T3σ

% --- Euclidean distances to reference trajectory ---
XR = XR(:,1:3); % XYZ coordinates
XR_ProjReal = ([XR(:,1)-Xo, XR(:,2)-Yo, XR(:,3)-Zo] * rotationMatrix);
XR_Projx_real = XR_ProjReal(:,1);
XR_Projy_real = XR_ProjReal(:,2);

XR_phase_real = atan2d(XR_Projy_real, XR_Projx_real) + 180;
XR_phase_real = round(XR_phase_real);
XR_phase_real(XR_phase_real == 360) = 0;

XR_distances = NaN(size(XR,1),1);
for i = 1:length(XR_phase_real)
    idx = XR_phase_real(i);
    if isnan(idx), continue; end
    XR_distances(i) = norm(XR(i,:) - RefTrajSmooth(idx+1,:));
end

% --- Classification by T1σ, T2σ, T3σ ---
nPoints = min(length(XR_distances), size(maxDeviation,1));
stabilityVector_01 = nan(size(XR_distances));
stabilityVector_01_02 = nan(size(XR_distances));
instabilityVector_02_03 = nan(size(XR_distances));
instabilityVector_03 = nan(size(XR_distances));

indices_ref = linspace(1,length(XR_distances),size(maxDeviation,1));
interpMax = interp1(indices_ref,maxDeviation(:,4),1:length(XR_distances));
interp2nd = interp1(indices_ref,maxDeviation(:,5),1:length(XR_distances));
interp3rd = interp1(indices_ref,maxDeviation(:,6),1:length(XR_distances));

%% --- Overall quantile threshold (baseline) ---

% 1) Define a baseline window before each disturbance
%    Here, we take the first disturbance and 60 seconds before as an example
baseline_sec   = round(perturbation_times);              % baseline duration (s)
pert1_idx      = round(perturbation_times(1) * fs);   % 1st perturbation index
b_start        = max(1, pert1_idx - baseline_sec*fs + 1);
b_end          = max(1, pert1_idx - 1);

% 2) r ratio = distance (point, reference trajectory) / effective_3sigma_radius
%    taken as the cumulative sum of the axes)
denom_all = interpMax + interp2nd + interp3rd;
denom_all = denom_all(:);
ratios    = XR_distances ./ denom_all;

% 3) Restrict to valid baseline points
ratios_base = ratios(b_start:b_end);
ratios_base = ratios_base(isfinite(ratios_base));

% 4) Select the overall quantile (e.g., 95% or 97.5% of baseline points will be stable)
q            = 0.975;                     % 0.95 or 0.975
r4_quantile  = quantile(ratios_base, q);  % Ratio threshold
resolution4  = max(1.0, r4_quantile);     % Never tighten (<1) the torus

% 5) (Optional) Logs for later tracing
if ~exist('logged_resolution4','var'), logged_resolution4 = []; end
if ~exist('baseline_unstable_pct_if_r4start','var'), baseline_unstable_pct_if_r4start = []; end
% Percentage of baseline instabilities if r4 = 1.0 :
unstable_mask_r4_1 = ratios_base > 1.0;
baseline_unstable_pct_if_r4start(end+1,1) = 100*mean(unstable_mask_r4_1);
logged_resolution4(end+1,1)               = resolution4;

for i = 1:length(XR_distances)
    T1 = resolution4 * (interpMax(i));
    T2 = resolution4 * (interpMax(i) + interp2nd(i));
    T3 = resolution4 * (interpMax(i) + interp2nd(i) + interp3rd(i));

    if XR_distances(i) <= T1
        stabilityVector_01(i) = XR_distances(i);
    elseif XR_distances(i) <= T2
        stabilityVector_01_02(i) = XR_distances(i);
    elseif XR_distances(i) <= T3
        instabilityVector_02_03(i) = XR_distances(i);
    else
        instabilityVector_03(i) = XR_distances(i);
    end
end

%% --- Percentages of T1o, T2o et T3o ---

nTotal = length(XR_distances(1:pert1_idx));

% Zones :
mask_T1o = XR_distances(1:pert1_idx) <= T1;                            % Very stable
mask_T2o = XR_distances(1:pert1_idx) > T1 & XR_distances(1:pert1_idx) <= T2;        % Stable
mask_T3o = XR_distances(1:pert1_idx) > T2 & XR_distances(1:pert1_idx) <= T3;        % Instable
mask_out = XR_distances(1:pert1_idx) > T3;                             % Very instable

% Percentages :
pct_T1o = 100 * sum(mask_T1o) / nTotal;
pct_T2o = 100 * sum(mask_T2o) / nTotal;
pct_T3o = 100 * sum(mask_T3o) / nTotal;
pct_out = 100 * sum(mask_out) / nTotal;

fprintf('\n--- Repartition ---\n');
fprintf('T1o (very stable)           : %.2f %%\n', pct_T1o);
fprintf('T2o (stable)                : %.2f %%\n', pct_T2o);
fprintf('T3o (instable)              : %.2f %%\n', pct_T3o);
fprintf('>T3o (very instable)        : %.2f %%\n', pct_out);

%% II. CoM recovery calculation

% Objective : detect the return of center of mass (CoM) to stable zone
% After each disturbance (fast & slow)

n_steps_window = 10;           % Minimum number of consecutive steps
recovery_Times = NaN(1,2);     % Recovery times [s]
recovery_indices = NaN(1,2);   % Frame indices of recovery
lag_time = NaN(1,2);           % Lag time [s]
peak_time = NaN(1,2);          % Peak time [s]
n_consecutive = 10;            % For lag_time (0.1 s if fs=100)
n_steps_recovery = NaN(1,2);   % Nombre de pas pour récupérer

for p = 1:2
    %% 1. Start of perturbation
    perturbation_start_idx = round(perturbation_times(p)*fs);
    if isnan(perturbation_start_idx) || perturbation_start_idx < 1 || perturbation_start_idx > length(XR_distances)
        continue;
    end

    %% 2. Pre-perturbation baseline
    if p == 1
        baseline_start_idx = 1;
        baseline_end_idx   = perturbation_start_idx;
    else
        baseline_start_idx = round((perturbation_times(1) + recovery_Times(1))*fs);
        baseline_end_idx   = round(perturbation_times(2)*fs);
    end
    if baseline_start_idx < 1, baseline_start_idx = 1; end
    if baseline_end_idx > length(XR_distances), baseline_end_idx = length(XR_distances); end
    if baseline_start_idx >= baseline_end_idx, continue; end

    baseline = XR_distances(baseline_start_idx:baseline_end_idx);
    mean_baseline = mean(baseline,'omitnan');
    std_baseline  = std(baseline,'omitnan');

    %% 3. Peak deviation & recovery calculation
    search_range_peak = perturbation_start_idx : ...
        min(length(XR_distances), perturbation_start_idx + round(5*fs));

    % Find all the peaks in the zone
    [all_peaks, all_locs] = findpeaks(XR_distances(search_range_peak));

    Frame_startRecovery = NaN;

    if ~isempty(all_peaks)
        for i = 1:length(all_peaks)
            peak_val = all_peaks(i);
            peak_idx_rel = all_locs(i); % relative to 'search_range_peak'
            candidate_idx = search_range_peak(1) + peak_idx_rel - 1;

            % Window of a certain number of frames after the peak
            window_end = min(length(XR_distances), candidate_idx + 700);
            post_window = XR_distances(candidate_idx:window_end);

            % Check if a point exceeds the peak
            if all(post_window(1) >= post_window(2:end))
                Frame_startRecovery = candidate_idx;
                break; % we keep the first peak that complies with the rule
            end
        end
    end

    % If no peak matches, fallback = classic maximum
    if isnan(Frame_startRecovery)
        [~, local_peak_idx] = max(XR_distances(search_range_peak));
        Frame_startRecovery = search_range_peak(1) + local_peak_idx - 1;
    end

    %% 4. Search recovery index based on step logic
    recovery_idx = NaN;
    tolerance_NaN = 5;   % maximum tolerance for NaN
    n_steps_window = 10;  % number of consecutive steps required

    % Converts timestamps to frames
    timestamps_frames = Peak; % Peak or round(timestamps_probe * fs)

    % Search loop from the peak
    for k = Frame_startRecovery:length(stabilityVector_01_02)

        % 1) First non-NaN point in stabilityVector_01_02
        if isnan(stabilityVector_01_02(k)), continue; end

        % 2) Identify the next 6 steps after k
        next_steps = timestamps_frames(timestamps_frames > k);
        if numel(next_steps) < n_steps_window
            break; % not enough steps after disturbance
        end
        step6 = next_steps(n_steps_window); % 6th steps

        % 3) Check stability
        range_check = k:step6;
        nan_count = sum(isnan(stabilityVector_01(range_check)) & ...
                        isnan(stabilityVector_01_02(range_check)));

        if nan_count <= tolerance_NaN
            recovery_idx = k;
            break; % found
        end
    end

    %% 5. Save if recovery point found

    if ~isnan(recovery_idx)
        recovery_indices(p) = recovery_idx;
        recovery_Times(p)   = (recovery_idx - Frame_startRecovery)/fs;
    else
        recovery_indices(p) = NaN;
        recovery_Times(p)   = NaN;
    end

    if ~isnan(recovery_idx)
        recovery_indices(p) = recovery_idx;
        recovery_Times(p)   = (recovery_idx - Frame_startRecovery)/fs;

        % --- Number of steps to recover stability ---
        steps_after_peak = timestamps_frames(timestamps_frames >= Frame_startRecovery & ...
                                             timestamps_frames <= recovery_idx);
        n_steps_recovery(p) = numel(steps_after_peak);
    else
        recovery_indices(p) = NaN;
        recovery_Times(p)   = NaN;
        n_steps_recovery(p) = NaN;
    end

    %% 6. Lag time calculation

    lag_time_idx = NaN;

    for k = perturbation_start_idx:length(instabilityVector_02_03)-n_consecutive+1
        window1 = instabilityVector_02_03(k:k+n_consecutive-1);
        window2 = instabilityVector_03(k:k+n_consecutive-1);

        % Creates a combined non-NaN mask
        combined_mask = ~isnan(window1) | ~isnan(window2);

        % If 10 consecutive instability values, take the first index
        if all(combined_mask)
            lag_time_idx = k;
            break;
        end
    end

    if ~isnan(lag_time_idx)
        lag_time(p) = (lag_time_idx - perturbation_start_idx)/fs;
    end

    %% 7. Peak time calculation

    if ~isnan(lag_time_idx) && ~isnan(Frame_startRecovery)
        peak_time(p) = (Frame_startRecovery - lag_time_idx)/fs;
    end

    %% 8. Secure graphical display
    figure;
    local_window = 2000;
    x_start = max(1, perturbation_start_idx-local_window);
    x_end   = min(length(XR_distances), perturbation_start_idx+3000);
    plot(x_start:x_end, XR_distances(x_start:x_end), 'Color',[0.4039 0.5372 0.6314],'LineWidth',1.2); hold on;
    plot(perturbation_start_idx, XR_distances(perturbation_start_idx), 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'c');
    plot(lag_time_idx, XR_distances(lag_time_idx), 's', 'MarkerSize', 8, 'MarkerEdgeColor', 'm', 'MarkerFaceColor', 'm');
    plot(stabilityVector_01, '-', 'Color',[229/255 78/255 209/255],'LineWidth',1.2);
    plot(stabilityVector_01_02, '-', 'Color',[237/255 179/255 20/255],'LineWidth',1.2);
    plot(instabilityVector_02_03, '-', 'Color',[191/255 216/255 52/255],'LineWidth',1.2);
    plot(instabilityVector_03, '-', 'Color',[0/255 7/255 97/255],'LineWidth',1.2);
    plot(Frame_startRecovery, XR_distances(Frame_startRecovery), '^', 'MarkerSize',8,'MarkerEdgeColor','r','MarkerFaceColor','r');
    if ~isnan(recovery_indices(p))
        plot(recovery_indices(p), XR_distances(recovery_indices(p)), 'o', 'MarkerSize',8,'MarkerEdgeColor','g','MarkerFaceColor','g');
    end
    legend('Distances', 'Perturbation', 'Lag time','T1σ','T2σ','T3σ','>T3σ','Max deviation','Recovery');
    title(['COM recovery after perturbation #' num2str(p)]);
    xlabel('Time (frames)'); ylabel('Euclidean distance (mm)');
    xlim([x_start x_end]); hold off;
end

disp('Recovery times (seconds) for each perturbation:');
disp(recovery_Times);

disp('Number of steps for recovery for each perturbation:');
disp(n_steps_recovery);

% recovery_Times = [n_steps_recovery(1), n_steps_recovery(2)]; % In steps
recovery_Times = [recovery_Times(1), recovery_Times(2)]; % In seconds

%% IV. Persistent difference - in steps

% cadence = round(length(timestamps_probe_valid) / 8);
% 
% baseline_duration_steps = 10;   % Baseline duration in steps (10 - 15)
% post_windows_steps = [10, 30, 50, cadence]; % Post-disturbance windows [steps]
% 
% % Pre-allocation (2 disturbances x 4 windows)
% persistent_diff     = zeros(2, length(post_windows_steps));
% persistent_diff_pct = zeros(2, length(post_windows_steps));
% baseline_mean       = zeros(1, 2);
% post_mean           = zeros(2, length(post_windows_steps));
% 
% for p = 1:2
%     % --- Baseline indices ---
%     perturb_idx       = round(perturbation_times(p) * fs);
%     baseline_start_idx = round(timestamps_frames(perturbation_indices(p) - 10));
%     baseline_end_idx   = perturb_idx - 1;
% 
%     % --- Baseline mean ---
%     data_pre = XR_distances(baseline_start_idx : baseline_end_idx);
%     baseline_mean(p) = mean(data_pre, 'omitnan');
% 
%     % --- Post-disturbance averages ---
%     for w = 1:length(post_windows_steps)
%         duration_pts   = post_windows_steps(w) + find(timestamps_frames >= recovery_indices(p), 1, 'first');
%         post_start_idx = recovery_indices(p);
%         post_end_idx = timestamps_frames(duration_pts);
% 
%         data_post = XR_distances(post_start_idx:post_end_idx - 1);
%         post_mean(p,w) = mean(data_post, 'omitnan');
% 
%         % Persistent differences (raw value and %)
%         persistent_diff(p,w)     = post_mean(p,w) - baseline_mean(p);
%         persistent_diff_pct(p,w) = 100 * persistent_diff(p,w) / baseline_mean(p);
%     end
% 
%     % % --- Console display ---
%     fprintf('Perturbation %d :\n', p);
%     for w = 1:length(post_windows_steps)
%         fprintf('  Post %2d steps : Persistence difference = %.4f, Diff %% = %.2f%%\n', ...
%             post_windows_steps(w), persistent_diff(p,w), persistent_diff_pct(p,w));
%     end
% end

%% IV. Persistent difference - in seconds

[~ ,loops] = findpeaks(DATA.Sacrum.filtered_value(:,2)); % Number of loops in the session
baseline_duration_sec = round(mean(diff(loops))) / 100; % Mean time of a loop
post_windows_sec = [baseline_duration_sec, baseline_duration_sec * 2, baseline_duration_sec * 3, baseline_duration_sec * 4]; % Windows

% baseline_duration_sec = 15;   % Baseline duration in seconds (15s - 1min)
% post_windows_sec = [15, 30, 45, 60]; % Post-disturbance windows [s]

% Pre-allocation (2 disturbances x 4 windows)
persistent_diff     = zeros(2, length(post_windows_sec));
persistent_diff_pct = zeros(2, length(post_windows_sec));
baseline_mean       = zeros(1, 2);
post_mean           = zeros(2, length(post_windows_sec));

%% V. Stacked windows (e.g., 1500, 3000, 4500, 6000)

for p = 1:2
    % --- Baseline indices ---
    perturb_idx       = round(perturbation_times(p) * fs);
    baseline_start_idx = max(1, perturb_idx - baseline_duration_sec * fs);
    baseline_end_idx   = perturb_idx - 1;

    % --- Baseline mean ---
    data_pre = XR_distances(baseline_start_idx : baseline_end_idx);
    baseline_mean(p) = mean(data_pre, 'omitnan');

    % --- Post-disturbance averages ---
    for w = 1:length(post_windows_sec)
        duration_pts   = post_windows_sec(w) * fs;
        post_start_idx = recovery_indices(p);
        post_end_idx   = min(length(XR_distances), post_start_idx + duration_pts);

        data_post = XR_distances(post_start_idx : post_end_idx);
        post_mean(p,w) = mean(data_post, 'omitnan');

        % Persistent differences (raw value and %)
        persistent_diff(p,w)     = post_mean(p,w) - baseline_mean(p);
        persistent_diff_pct(p,w) = 100 * persistent_diff(p,w) / baseline_mean(p);
    end

    % % --- Console display ---
    fprintf('Perturbation %d :\n', p);
    for w = 1:length(post_windows_sec)
        fprintf('  Post %2d s : Persistence difference = %.4f, Diff %% = %.2f%%\n', ...
            post_windows_sec(w), persistent_diff(p,w), persistent_diff_pct(p,w));
    end
end

%% V. Non-stacked windows (e.g., 1500, 1500, 1500, 1500)

for p = 1:2
    % --- Baseline indices ---
    perturb_idx       = round(perturbation_times(p) * fs);
    baseline_start_idx = max(1, perturb_idx - baseline_duration_sec * fs);
    baseline_end_idx   = perturb_idx - 1;

    % --- Baseline mean ---
    data_pre = XR_distances(baseline_start_idx : baseline_end_idx);
    baseline_mean(p) = mean(data_pre, 'omitnan');

    for w = 1:length(post_windows_sec)
        if w == 1
            duration_pts   = post_windows_sec(w) * fs;
            post_start_idx = recovery_indices(p);
            post_end_idx   = min(length(XR_distances), post_start_idx + duration_pts - 1);

        else
            post_start_idx = post_end_idx + 1;
            post_end_idx   = min(length(XR_distances), post_start_idx + duration_pts) - 1;
        end

        % ---- DEBUG PRINT ----
        fprintf('Perturbation %d | Window %d : start=%d, end=%d, duration=%d pts\n', ...
                p, w, post_start_idx, post_end_idx, post_end_idx - post_start_idx + 1);

        % ---- Compute values ----
        data_post = XR_distances(post_start_idx : post_end_idx);
        post_mean(p,w) = mean(data_post, 'omitnan');

        persistent_diff(p,w)     = post_mean(p,w) - baseline_mean(p);
        persistent_diff_pct(p,w) = 100 * persistent_diff(p,w) / baseline_mean(p);
    end
end

                                    %% --- 3D Phase Space Reconstruction ---

% --- Graphical parameters ---
mainColor   = [0.6 0.2 0.8]; % purple
pointColor  = 'ro'; % center
ellipseCols = {[237 179 20]/255, [255 102 0]/255, [200 0 0]/255}; % yellow, orange, red
resPlusOne  = resolution + 1;

% --- Visualization for the 2 perturbations ---
plotPhaseSpace(round(perturbation_times(1) * 100): recovery_indices(1), 'Phase Space Reconstruction : 1st perturbation', ...
               RefTrajSmooth, Xo, Yo, Zo, ellipses2, ...
               Original, Delayed1, Delayed2, ...
               resPlusOne, ellipseCols, mainColor, pointColor);

plotPhaseSpace(round(perturbation_times(2) * 100): recovery_indices(2), 'Phase Space Reconstruction : 2nd perturbation', ...
               RefTrajSmooth, Xo, Yo, Zo, ellipses2, ...
               Original, Delayed1, Delayed2, ...
               resPlusOne, ellipseCols, mainColor, pointColor);

                                    %% --- Save results ---
% --- Participant's ID ---
participantID = str2double(regexp(fileName, '^\d+', 'match', 'once'));

% --- Group ---
if ismember(participantID, young_IDs)
    group = "Young";
elseif ismember(participantID, elderly_IDs)
    group = "Aging";
else
    warning('Unknown group for participant %d', participantID);
    group = "Unknown";
end

% --- Discharge folder ---
outputFolder = fullfile('..., RES'); %% Insert your file path
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% ----------  1. Recovery / Resynchronisation ----------
if find(slow_first == participantID)
    perturbation = ["Slow"; "Slow"; "Fast"; "Fast"];
else
    perturbation = ["Fast"; "Fast"; "Slow"; "Slow"];
end

Tnew = table( ...
    repmat(participantID,4,1), ...
    repmat(group,4,1), ...
    perturbation, ...
    ["COM"; "Synchronization"; "COM"; "Synchronization"], ...
    [recovery_Times(1); resync_times(1); recovery_Times(2); resync_times(2)], ...
    'VariableNames', {'Participant','Group','Perturbation','Measure','Value'});

appendToCSV(fullfile(outputFolder,'CoM_Synchronization.csv'), Tnew);

%% ---------- 2. Resilience parameters ----------

Tnew = table(...
    repmat(participantID,4,1), ...
    repmat(group,4,1), ...
    perturbation, ...
    ["Lag time"; "Peak time"; "Lag time"; "Peak time"], ...
    [lag_time(1); peak_time(1); lag_time(2); peak_time(2)], ...
    'VariableNames', {'Participant', 'Group', 'Perturbation', 'Measure', 'Value'});

appendToCSV(fullfile(outputFolder, 'Resilience.csv'), Tnew);

%% ---------- 2. Persistence difference ----------

if find(slow_first == participantID)
    perturb = ["Slow"; "Slow"; "Slow"; "Slow"; "Slow"; ...
               "Fast"; "Fast"; "Fast"; "Fast"; "Fast"];
else
    perturb = ["Fast"; "Fast"; "Fast"; "Fast"; "Fast"; ...
               "Slow"; "Slow"; "Slow"; "Slow"; "Slow"];
end

Tnew = table( ...
    repmat(participantID,10,1), ...
    repmat(group,10,1), ...
    perturb, ...
    ["Baseline1";"Pers1-15s";"Pers1-30s";"Pers1-45s";"Pers1-60s"; ...
     "Baseline2";"Pers2-15s";"Pers2-30s";"Pers2-45s";"Pers2-60s"], ...
    [baseline_mean(1); post_mean(1,1:4)'; baseline_mean(2); post_mean(2,1:4)'], ...
    'VariableNames', {'Participant','Group','Perturbation','Measure','Value'});

appendToCSV(fullfile(outputFolder,'persistence_difference.csv'), Tnew);

%% ---------- 3. Percentage of T1σ, T2σ, T3σ, >T3σ ----------

Tnew = table( ...
    repmat(participantID,4,1), ...
    repmat(group,4,1), ...
    ["T1o"; "T2o"; "T3o"; ">T3o"], ...
    [pct_T1o; pct_T2o; pct_T3o; pct_out], ...
    'VariableNames', {'Participant','Group','Measure','Value'});

appendToCSV(fullfile(outputFolder,'percentage_stability.csv'), Tnew);

                                    %% --- Functions ---

%% --------- For 3D phase space reconstruction ----------
function plotPhaseSpace(range, titleStr, RefTrajSmooth, Xo, Yo, Zo, ...
                        ellipses2, Original, Delayed1, Delayed2, ...
                        resPlusOne, ellipseCols, mainColor, pointColor)

    figure; hold on; grid on;

    % Reference trajectory
    plot3(RefTrajSmooth(:,1), RefTrajSmooth(:,2), RefTrajSmooth(:,3), ...
        'LineWidth', 2, 'Color', mainColor);

    % Centroid
    scatter3(Xo, Yo, Zo, pointColor, 'filled');

    % Ellipses
    for j = 1:3:resPlusOne*3
        plot3(ellipses2(:,j), ellipses2(:,j+1), ellipses2(:,j+2), ...
            'Color', ellipseCols{1}, 'LineWidth', 0.5);
    end

    % Disturbed segment
    plot3(Original(range), Delayed1(range), Delayed2(range), ...
        'k', 'LineWidth', 0.3);

    % Formatting
    title(titleStr);
    xlabel('X(t)'); ylabel('X(t+\tau)'); zlabel('X(t+2\tau)');
    view(3); camlight; lighting gouraud;
end

%% ----------- For csv save ----------
function appendToCSV(outputFile, Tnew)
    if isfile(outputFile)
        Told = readtable(outputFile, 'Delimiter',';');

        % Delete existing participants rows
        Told(Told.Participant == Tnew.Participant(1), :) = [];

        % Harmonize types
        Told.Group = string(Told.Group);
        Tnew.Group = string(Tnew.Group);

        % Merge and sort
        Tsorted = sortrows([Told; Tnew], 'Participant');

        % Complete rewrite
        writetable(Tsorted, outputFile, 'Delimiter',';');
    else
        % New file
        writetable(Tnew, outputFile, 'Delimiter',';');
    end
end