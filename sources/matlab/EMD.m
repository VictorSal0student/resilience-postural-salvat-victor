clear
close all
clc

%% Open the file containing data

cd '...\DATA\RAW'; % Insert your file path
DATA = load("040_Cued.mat"); % Load the file in the folder pre-selected
DATA = DATA.pMOCAP.MOCAP;

%%

keepFields = {'MFilter', 'Analog'}; % Fields to keep
allFields = fieldnames(DATA); % Full list

% Retiring every fields except 'Marker' & 'Analog'

for i = 1:numel(allFields)
    if ~ismember(allFields{i}, keepFields)
        DATA = rmfield(DATA, allFields{i});
    end
end

AllData = DATA;

DATA = DATA.MFilter;

%% Adjustments

fieldNames = fieldnames(DATA);  % Get the field names of the struct

% Iterate over each field
for i = 1:length(fieldNames)
    fieldName = fieldNames{i};

    % Check if the field contains a 'value' field
    if isfield(DATA.(fieldName), 'value')
        % Get the data from the 'value' field
        data = DATA.(fieldName).value;

        % Remove only the first point
        data = data(2:end, :);

        % Subtract the mean from the data
        data = data - mean(data);

        % Store the modified data back into the struct
        DATA.(fieldName).value = data;
    end
end

% fieldNames = fieldnames(DATA);  % Get the field names of the struct
% 
% remove_first = 1 * 400;  % Number of samples to remove from the start
% remove_end = 1 * 400;    % Number of samples to remove from the end
% 
% fieldNames = fieldnames(DATA);  % Get the field names of the struct
% 
% % Iterate over each field
% for i = 1:length(fieldNames)
%     fieldName = fieldNames{i};
% 
%     % Check if the field contains a  'value' field (to avoid errors)
%     if isfield(DATA.(fieldName), 'value')
%         % Get the data from the 'value' field
%         data = DATA.(fieldName).value;
% 
%         % Remove data at the beginning and the end
%         data = data(remove_first:end - remove_end, :);
% 
%         % Subtract the mean from the data
%         data = data - mean(data);
% 
%         % Store the modified data back into the struct
%         DATA.(fieldName).value = data;
%     end
% end

%% Assembling markers

data_matrix = [DATA.Dos01.value, DATA.Dos02.value, DATA.Dos03.value, DATA.Dos04.value];

% Calculate barycenter for each row across the four markers
% Mean of columns for x, y, and z coordinates
mean_x = mean(data_matrix(:, [1, 4, 7, 10]), 2);
mean_y = mean(data_matrix(:, [2, 5, 8, 11]), 2);
mean_z = mean(data_matrix(:, [3, 6, 9, 12]), 2);

Silence_data = [mean_x, mean_y, mean_z]; % Unified data

%% Time the data

fs = 400; % Sampling frequency

Seconds = length(mean_x) ./ fs;
Time = zeros(length(mean_x),1);

for i = 1:length(mean_x)
    Time(i) = i ./ length(mean_x) .* Seconds;
end

%% Application of EMD

[imf,~,~] = emd(mean_z, Interpolation = "pchip"); %% Select the dataset for EMD

% Plot original signal and IMFs
figure;
subplot(size(imf,2)+1,1,1);
plot(Time, mean_z);
title('Original Signal');
for k = 1:size(imf,2)
    subplot(size(imf,2)+1,1,k+1);
    plot(Time, imf(:,k));
    title(['IMF ' num2str(k)]);
    ylim([-50 50])
end

disp(min(imf(:,1)));
disp(max(imf(:,1)));

%% Assemble imf values to have a correct signal

% newdata = imf(:,1) + imf(:,6) + imf(:,7) + imf(:,8) + imf(:,9) + imf(:,10); % For 'Marker'
newdata = imf(:,4) + imf(:,5) + imf(:,6) + imf(:,7) + imf(:,8) + imf(:,9) + imf(:,10); % For 'MFilter'
imf_values = mean_z - newdata;

figure;
subplot(3,1,1)
plot(Time, mean_z)
subplot(3,1,2)
plot(Time, newdata)
subplot(3,1,3)
plot(Time, imf_values)

%% Low pass filter

DATA.Sacrum.value = [mean_x, mean_y, imf_values];

fs = DATA.Dos01.Rate; % Sampling frequency

forder = 4;
cutfreq = 5;

% Create the filter coefficients
[b, a] = butter(forder, cutfreq / fs);

fieldNames = fieldnames(DATA);  % Get the field names of the struct

% Iterate over each field
for i = 1:length(fieldNames)
    fieldName = fieldNames{i};

    % Check if the field contains a 'value' field (to avoid errors)
    if isfield(DATA.(fieldName), 'value')
        data = DATA.(fieldName).value;

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

%% Downsampling

% downsampleFactor = 4;
% fieldNames = fieldnames(DATA);
% 
% for i = 1:length(fieldNames)
%     fieldName = fieldNames{i};
% 
%     if isfield(DATA.(fieldName), 'filtered_value')
%         data = DATA.(fieldName).filtered_value;
% 
%         % Preallocate output
%         [rows, cols] = size(data);
%         new_rows = floor(rows / downsampleFactor);
%         downsampled = zeros(new_rows, cols);
% 
%         % Apply decimate column-wise
%         for c = 1:cols
%             temp = decimate(data(:, c), downsampleFactor);
%             downsampled(1:length(temp), c) = temp;  % assign downsampled data
%         end
% 
%         % Store result in a new field
%         DATA.(fieldName).filtered_value = downsampled;
%     end
% end
% 
%% Downsampling

downsampleFactor = 4;
fieldNames = fieldnames(DATA);

for i = 1:length(fieldNames)
    fieldName = fieldNames{i};

    if isfield(DATA.(fieldName), 'value')
        data = DATA.(fieldName).value;

        % Preallocate output
        [rows, cols] = size(data);
        new_rows = floor(rows / downsampleFactor);
        downsampled = zeros(new_rows, cols);

        % Apply decimate column-wise
        for c = 1:cols
            temp = decimate(data(:, c), downsampleFactor);
            downsampled(1:length(temp), c) = temp;  % assign downsampled data
        end

        % Store result in a new field
        DATA.(fieldName).downsampled_value = downsampled;
    end
end

DATA.Analog = AllData.Analog.Analog01;

save('...\DATA\IMF\040_Cued_imf.mat', 'DATA'); % Insert your file path & adapt the name