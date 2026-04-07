function ami = AMI_function(data, delay)

% AMI_function : Compute the Average Mutual Information (AMI) for a time series
%
%   Inputs :
%       data  : Time series (vector, real values). It will be reshaped into column
%       delay : Time delay (positive integer < length(data))
%
%   Outputs :
%       ami   : Estimated average mutual information valu for the given delay
%
%   Description :
%       The function computes the mutual information between the signal
%       and a delayed version of itself using a histogram-based probability
%       density estimation.
%
%   Steps :
%       1. Ensure input validity (vector + delay check)
%       2. Build delayed time series : X(t) vs X(t+delay)
%       3. Estimate joint and marginal probabilities with histograms
%       4. Compute Average Mutual Information (AMI) as :
%           AMI = sum(P(x,y) * log(P(x,y) / P(x,y) / (P(x)*P(y))))

    % --- Preprocessing ---
    data = data(:); % Force column vector

    % Check for empty input
    if isempty(data)
        error('Input data is empty.');
    end

    % Define histogram edges
    edges = linspace(min(data), max(data), 50);
    
    % Number of data points
    N = length(data);

    % Validate delay
    if delay >= N
        error('Delay is too large for the dataset size.');
    end
    
    % --- Build delay data ---
    X = data(1:N-delay); % Original values
    Y = data(1+delay:N); % Delayd values

    % --- Joint histogram ---
    [hist2D, ~, ~] = histcounts2(X, Y, edges, edges);

    % Normalize to obtain joint probability distribution
    jointProb = hist2D / sum(hist2D, 'all');

    % Marginal probabilities
    marginalProbX = sum(jointProb, 2); % sum over Y
    marginalProbY = sum(jointProb, 1); % sum over X

    % --- Compute AMI ---
    ami = 0;
    for i = 1:length(edges)-1
        for j = 1:length(edges)-1
            if jointProb(i, j) > 0
                ami = ami + jointProb(i, j) * log(jointProb(i, j) / ...
                    (marginalProbX(i) * marginalProbY(j)));
            end
        end
    end
end
