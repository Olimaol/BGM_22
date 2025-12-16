function [hrf,p] = spm_hrf(RT)
% Haemodynamic response function
% FORMAT [hrf,p] = spm_hrf(RT)
% RT   - scan repeat time
% p    - parameters of the response function (two Gamma functions)
%
%                                                           defaults
%                                                          {seconds}
%        p(1) - delay of response (relative to onset)          6
%        p(2) - delay of undershoot (relative to onset)       16
%        p(3) - dispersion of response                         1
%        p(4) - dispersion of undershoot                       1
%        p(5) - ratio of response to undershoot                6
%        p(6) - onset {seconds}                                0
%        p(7) - length of kernel {seconds}                    32
%
% T    - microtime resolution [Default: 16]
%
% hrf  - haemodynamic response function
% p    - parameters of the response function
%__________________________________________________________________________
%
% The parameters p(1:4) correspond to the shape and scale parameters of two
% probability density functions of the Gamma distribution (see spm_Gpdf.m),
% one corresponding to the main response and the other one to the
% undershoot.
% Note that the mean of the Gamma distribution is shape*scale and its mode
% is (shape-1)*scale.  This means that with the default values of the
% parameters the peak of the heamodynamic response function will be around
% 5 seconds.
%__________________________________________________________________________

% Karl Friston
% Copyright (C) 1996-2022 Wellcome Centre for Human Neuroimaging

% Modified 2025 Oliver Maith - only use default parameters


%-Hardcoded default parameters
%--------------------------------------------------------------------------
p = [6 16 1 1 6 0 32];
fMRI_T = 16;

%-Modelled haemodynamic response function - {mixture of Gammas}
%--------------------------------------------------------------------------
dt  = RT/fMRI_T;
u   = [0:ceil(p(7)/dt)] - p(6)/dt;
hrf = spm_Gpdf(u,p(1)/p(3),dt/p(3)) - spm_Gpdf(u,p(2)/p(4),dt/p(4))/p(5);
hrf = hrf([0:floor(p(7)/RT)]*fMRI_T + 1);
hrf = hrf'/sum(hrf);
