function [stdx,stdy,stdz] = fct_std_RefTraj(PreOriginalS,phaseAngleRef,PreDataPhaseAngleRS,NewxpPre,NewypPre,NewzpPre,resolution2)

% Find standard deviation at each phase angle
%Input:
%   PreOriginalS        - Vector containing pre-perturbation sorted dataset of "x-axis"
%   phaseAngleRef       - Vector with resolution3 number of phase angles
%   PreDataPhaseAngleRS - Matrix with info on each datapoint:
%                         [PreOriginalS, PreDelayedS1, PreDelayedS2, PhaseAngleRS]
%   Original            - Vector containing complete dataset of "x-axis"
%   NewxpPre            - Vector with Values of fitted reference trajectory
%                         for each phase angle, x-axis
%   NewypPre            - Vector with Values of fitted reference trajectory
%                         for each phase angle, y-axis
%   NewzpPre            - Vector with Values of fitted reference trajectory
%                         for each phase angle, z-axis
%   resolution2         - resolution for splitting phaseangles
% 
%
%Output:
%   stdx, stdy, stdz    - Vector with std for every phaseangle
%
% Contributed authors: Ravi Deepak (depakroshanblu@gmail.com), Marc Bartholet, Caroline Heimhofer
% Affiliation: Laboratory of Movement Biomechanics, ETH Zurich, Switzerland
% Last Modified: June 2019

k = 1;
n = 1;
stdx = zeros(1,length(phaseAngleRef));
stdy = zeros(1,length(phaseAngleRef));
stdz = zeros(1,length(phaseAngleRef));
for i = 1:length(PreOriginalS)
    if phaseAngleRef(k) == PreDataPhaseAngleRS(i,4)
        stdx(k) = std_general(PreDataPhaseAngleRS(n:i,1),NewxpPre(k),length(PreDataPhaseAngleRS(n:i,1)));
        stdy(k) = std_general(PreDataPhaseAngleRS(n:i,2),NewypPre(k),length(PreDataPhaseAngleRS(n:i,2)));
        stdz(k) = std_general(PreDataPhaseAngleRS(n:i,3),NewzpPre(k),length(PreDataPhaseAngleRS(n:i,3)));
        
    else
        if k >= resolution2
            break;
        else
            if phaseAngleRef(k+1) == PreDataPhaseAngleRS(i,4)
                k = k + 1;
                n = i;
                stdx(k) = std_general(PreDataPhaseAngleRS(n:i,1),NewxpPre(k),length(PreDataPhaseAngleRS(n:i,1)));
                stdy(k) = std_general(PreDataPhaseAngleRS(n:i,2),NewypPre(k),length(PreDataPhaseAngleRS(n:i,2)));
                stdz(k) = std_general(PreDataPhaseAngleRS(n:i,3),NewzpPre(k),length(PreDataPhaseAngleRS(n:i,3)));
                
            else
                k = k + 1;
                for k = k:resolution2
                    if phaseAngleRef(k) == PreDataPhaseAngleRS(i,4)
                        n = i;
                        stdx(k) = std_general(PreDataPhaseAngleRS(n:i,1),NewxpPre(k),length(PreDataPhaseAngleRS(n:i,1)));
                        stdy(k) = std_general(PreDataPhaseAngleRS(n:i,2),NewypPre(k),length(PreDataPhaseAngleRS(n:i,2)));
                        stdz(k) = std_general(PreDataPhaseAngleRS(n:i,3),NewzpPre(k),length(PreDataPhaseAngleRS(n:i,3)));
                        break;
                    end
                end
            end
        end
    end
end
end



