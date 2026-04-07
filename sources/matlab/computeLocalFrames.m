function [T, N, B] = computeLocalFrames(RefTrajSmooth)

% computeLocalFrames : Compute the local Frenet-Serret-like frames (T, N, B) along a 3D trajectory
%
%   Input :
%       RefTrajSmooth : [nPoints x 3] matrix, 3D reference trajectory (x, y, z)
%
%   Outputs :
%       T : [nPoints x 3] Tangent vectors at each point
%       N : [nPoints x 3] Nomrmal vectors (variation of tangent)
%       B : [nPoints x 3] Binormal vectors (cross product of T and N)
%
%   Description :
%       This function computes a discrete approximation of Frenet-Serret
%       frames along a smoothed 3D trajectory :
%       - Tangents are estimated by centered finite differences
%       - Normals are derived from tangent variations
%       - Binormals are given by cross T and N

    % --- Initialization ---
    nPoints = size(RefTrajSmooth,1);
    T = zeros(nPoints,3);
    N = zeros(nPoints,3);
    B = zeros(nPoints,3);

    % --- Tangent vectors ---
    % Use centered differences for interior points
    for i = 2:nPoints-1
        T(i,:) = RefTrajSmooth(i+1,:) - RefTrajSmooth(i-1,:);
        T(i,:) = T(i,:) / norm(T(i,:));
    end
    % Forward difference at start
    T(1,:) = RefTrajSmooth(2,:) - RefTrajSmooth(1,:);
    T(1,:) = T(1,:) / norm(T(1,:));
    % Backward difference at end
    T(end,:) = RefTrajSmooth(end,:) - RefTrajSmooth(end-1,:);
    T(end,:) = T(end,:) / norm(T(end,:));

    % --- Normal vectors ---
    % Variation of tangents
    for i = 2:nPoints-1
        dT = T(i+1,:) - T(i-1,:);
        N(i,:) = dT / norm(dT);
    end
    % At borders
    N(1,:) = T(2,:) - T(1,:);
    N(1,:) = N(1,:) / norm(N(1,:));
    N(end,:) = T(end,:) - T(end-1,:);
    N(end,:) = N(end,:) / norm(N(end,:));

    % --- Binormal vectors ---
    % Cross product of T and N
    for i = 1:nPoints
        B(i,:) = cross(T(i,:), N(i,:));
        B(i,:) = B(i,:) / norm(B(i,:));
    end
end