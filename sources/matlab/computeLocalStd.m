function [stdLocalN, stdLocalB, stdLocalT] = computeLocalStd(RefTrajSmooth, PreDataPhaseAngleRS, phaseAngleRef)

% computeLocalStd : Compute local standard deviations along a trajectory
% in the directions of the Frenet-Serret-like frames (T, N, B)
%
%   Inputs :
%       RefTrajSmooth       : [nPoints x 3] smoothed reference trajectory (x, y, z)
%       PreDataPhaseAngleRS : [m x 4] matrix of experimental points, with :
%                               cols 1:3 = coordinates of points
%                               col 4    = correponding phase angles
%       phaseAngleRef       : [resolution x 1] vector of reference phase angles
%
%   Outputs :
%       stdLocalN : [resolution x 1] local standard deviation along Normal
%       stdLocalB : [resolution x 1] local standard deviation along Binormal
%       stdLocalT : [resolution x 1] local standard deviation along Tangent
%
%   Description :
%       For each phase angle in the reference trajectory :
%           1. Find experimental points corresponding to that phase
%           2. Express local deviations in the Frenet-Serret-like frame (T, N, B)
%           3. Compute the standard deviation along each axis

    % --- Initialization ---
    resolution = length(phaseAngleRef);

    % Compute local frames (T, N, B)
    [T, N, B] = computeLocalFrames(RefTrajSmooth);

    % Allocate outputs
    stdLocalN = zeros(resolution,1);
    stdLocalB = zeros(resolution,1);
    stdLocalT = zeros(resolution,1);

    % --- Loop over phase angles ---
    for k = 1:resolution
        % Select experimental points with same phase angle
        indices = (PreDataPhaseAngleRS(:,4) == phaseAngleRef(k));

        if sum(indices) == 0
            % No data points for this phase -> assign NaN
            stdLocalN(k) = NaN;
            stdLocalB(k) = NaN;
            stdLocalT(k) = NaN;
            continue
        end

        % Center = Reference trajectory at this phase
        center = RefTrajSmooth(k,:);

        % Extract experimental points for this phase
        pts = PreDataPhaseAngleRS(indices, 1:3);

        % Local devuation vectors
        vecs = pts - center;

        % Projection onto local frame
        projN = vecs * N(k,:)';
        projB = vecs * B(k,:)';
        projT = vecs * T(k,:)';

        % Local standard deviations (NaNs ignored)
        stdLocalN(k) = std(projN, 'omitnan');
        stdLocalB(k) = std(projB, 'omitnan');
        stdLocalT(k) = std(projT, 'omitnan');
    end
end