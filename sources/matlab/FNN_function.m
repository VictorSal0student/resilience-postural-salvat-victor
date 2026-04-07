function fnnRate = FNN_function(data, embeddingDimension, delay)

%   FNN_function : Compute the False Nearest Neighbors (FNN) rate for time series embedding
%
%   Inputs :
%       data               : Time series (vector, real values)
%       embeddingDimension : Embedding dimension (integer > 1)
%       delay              : Time delay (integer > 0)
%
%   Output :
%       fnnRate            : Percentage of false nearest neighbors [%]
%
%   Description :
%       This function estimates the percentage of False Nearest Neighbors
%       (FNN) when embedding a time series into a higher-dimensional phase
%       space. It helps to determine the optimal embedding dimension for
%       phase-space reconstruction.
%
%   Steps :
%       1. Phase space reconstruction with delay embedding
%       2. Remove rows containing NaNs in the embedded space
%       3. Nearest-neighbor search in the embedding space
%       4. Compute FNN by checking whether distances increase drastically
%          when going to the next embedding dimension


    % --- Input preparation ---
    N = length(data); % Length of time series

    % --- Phase space reconstruction ---
    maxIdx = N - (embeddingDimension - 1) * delay;
    embeddedData = zeros(maxIdx, embeddingDimension);
    for i = 1:embeddingDimension
        embeddedData(:, i) = data((1:maxIdx) + (i - 1) * delay);
    end

    % --- Reomove invalid rows (containing NaNs) ---
    validRows = all(~isnan(embeddedData), 2);
    embeddedData = embeddedData(validRows, :);
    baseIndices = find(validRows); % Indices in original time series

    % Early exit if not enough valid points
    if size(embeddedData, 1) < 2
        warning('Pas assez de données valides pour le calcul FNN.');
        fnnRate = NaN;
        return;
    end

    % --- Nearest neighbor search ---
    Mdl = createns(embeddedData, 'Distance', 'euclidean');
    [nearestIdx, nearestDist] = knnsearch(Mdl, embeddedData, 'K', 2);

    % --- False Nearest Neighbors (FNN) calculation ---
    fnnCount = 0;
    validCount = 0;

    for i = 1:size(embeddedData, 1)
        neighborIdx = nearestIdx(i, 2);

        % Original indices in the raw time series
        idx_i = baseIndices(i);
        idx_n = baseIndices(neighborIdx);

        % Check bounds for embedding extension
        if (idx_i + embeddingDimension * delay > N) || ...
           (idx_n + embeddingDimension * delay > N)
            continue;
        end

        % Skip if extrapolated values are NaN
        if isnan(data(idx_i + embeddingDimension * delay)) || ...
           isnan(data(idx_n + embeddingDimension * delay))
            continue;
        end

        validCount = validCount + 1;

        % Distance increase when moving to next embedding dimension
        distIncreased = abs(data(idx_i + embeddingDimension * delay) - ...
                            data(idx_n + embeddingDimension * delay));

        % Criterion : distance increases by more than factor 10
        if (distIncreased / nearestDist(i, 2)) > 10
            fnnCount = fnnCount + 1;
        end
    end

    % --- Final FNN rate [%] ---
    fnnRate = 100 * fnnCount / validCount;
end