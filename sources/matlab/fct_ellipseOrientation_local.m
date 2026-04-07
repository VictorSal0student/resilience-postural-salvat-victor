function [ellipse3d] = fct_ellipseOrientation_local(RefTrajSmooth, stdLocalN, stdLocalB, N, B, resolution3)

% fct_ellipseOrientation_local : Generates local ellipses around a 3D trajectory
%
%   Inputs :
%       RefTrajSmooth : [Nx3] smoothed 3D trajectory (N points)
%       stdLocalN     : [Nx1] standard deviation along the local normal
%       stdLocalB     : [Nx1] standard deviation along the local binormal
%       N             : [Nx3] local normal vectors
%       B             : [Nx3] local binormal vectors
%       resolution3   : number of points to discretize each ellipse
%
%   Outputs :
%       ellipse3d : [resolution3 x (N*3)] ellipse coordinates
%                   (each block of 3 columns corresponds to an ellipse in (x,y,z)
%   Description :
%       This function constructs local ellipses around a 3D reference
%       trajectory using the local Frenet-Serret-like frame (Normal, Binormal).
%       For each point of the trajectory :
%           1. Define an ellipse in the (N,B) plane with radii equal to the
%              Local standard deviations (stdLocalN, stdLocalB)
%           2. Discretize it into 'resolution3' points
%           3. Translate it to the corresponding trajectory point
%           The result is a set of ellipses aligned with the local geometry
%           of the trajectory, useful for visualizing variability or uncertainty.


    % Number of points on trajectory
    resolution2 = size(RefTrajSmooth, 1);

    % Pre-allocate final table
    ellipse3d = zeros(resolution3, resolution2 * 3);

    % Angular parameterization of the ellipse
    theta = linspace(0, 2*pi, resolution3);

    % Loop on every point of the trajectory
    for i = 1:resolution2
        % Ellipse axes (N = normal, B = binormal)
        a = stdLocalN(i); % major axe
        b = stdLocalB(i); % minor axe

        % Parametric equation of the ellipse in the local plane (N, B)
        ellipse = (a * cos(theta))' * N(i,:) + (b * sin(theta))' * B(i,:);
        
        % Translation at the current point of the trajectory
        ellipse3D = ellipse + RefTrajSmooth(i,:);

        % Storage of coordinates (block of 3 columns : x, y, z)
        j = (i-1)*3 + 1;
        ellipse3d(:,j) = ellipse3D(:,1);
        ellipse3d(:,j+1) = ellipse3D(:,2);
        ellipse3d(:,j+2) = ellipse3D(:,3);
    end
end