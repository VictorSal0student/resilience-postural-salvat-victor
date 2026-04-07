function [std] = std_general(x,xref,n)

% std_general : Compute the standard deviation relative to a reference value.
%
%   Inputs :
%       x    : Vector of values
%       xref : Reference value (mean, median, or custom)
%       n    : Number of samples (used in denominator)
%
%   Output :
%       s    : Standard deviation relative to xref
%
%   Formula :
%       s = sqrt(sum(x-xref).^2) / (n-1))

std = 0;
for i = 1:length(x)
    std = std + (x(i)-xref)^2; % squared differences
end
std = sqrt(std/(n-1));
end