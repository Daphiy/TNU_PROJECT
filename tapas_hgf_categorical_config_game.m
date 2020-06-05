
function c = tapas_hgf_categorical_config_game

% Tips:
% - When analyzing a new dataset, take your inputs u and use
%
%   >> est = tapas_fitModel([], u, 'tapas_hgf_categorical_config', 'tapas_bayes_optimal_categorical_config');
%
%   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
%   this file here, so choose them wide and loose to let the inputs influence the result). You can
%   then use the optimal parameters as your new prior means for the perceptual parameters.
%
% - If you get an error saying that the prior means are in a region where model assumptions are
%   violated, lower the prior means of the omegas, starting with the highest level and proceeding
%   downwards.
%
% - Alternatives are lowering the prior mean of kappa, if they are not fixed, or adjusting
%   the values of the kappas or omegas, if any of them are fixed.
%
% - If the log-model evidence cannot be calculated because the Hessian poses problems, look at
%   est.optim.H and fix the parameters that lead to NaNs.
%
% - Your guide to all these adjustments is the log-model evidence (LME). Whenever the LME increases
%   by at least 3 across datasets, the adjustment was a good idea and can be justified by just this:
%   the LME increased, so you had a better model.

c = struct;

% Model name
c.model = 'hgf_categorical';

% Number of states
c.n_outcomes = 3;

% Upper bound for kappa and theta (lower bound is always zero) - changint
% this you have different responces!
c.kaub = 0.05*2; %1 makes more sence, brakes at 3, 0: zero volatility, different results for 2! HIGHER PUSHES VOLATILITY UP
c.thub = 0.400*2; %0.1, at 0 it changes like a low sensory uncertanty - lovering it down sems to give a lower volatility 
%pushing tetha up we have an higher change in volatility
% Sufficient statistics of Gaussian parameter priors

% Initial mu2
c.mu2_0mu = repmat(tapas_logit(1/c.n_outcomes,1),1,c.n_outcomes); %don't change
c.mu2_0sa = zeros(1,c.n_outcomes); %don't change

% Initial sigma2
c.logsa2_0mu = repmat(log(1),1,c.n_outcomes);%don't change
c.logsa2_0sa = zeros(1,c.n_outcomes);%don't change

% Initial mu3
% Usually best kept fixed to 1 (determines origin on x3-scale).
c.mu3_0mu = 1; 
c.mu3_0sa = 0; %0

% Initial sigma3
c.logsa3_0mu = log(0.1); %0.1
c.logsa3_0sa = 1; %1

% Kappa
% This should be fixed (preferably to 1) if the observation model
% does not use mu3 (kappa then determines the scaling of x3).
c.logitkamu = 0; % 1
c.logitkasa = 0; % this is 0, and c.kaub = 2 above, then kappa is fixed to 1

% Omega
c.ommu =  -0.5095; %-2
c.omsa = 3^2; %5^2

% Theta
c.logitthmu = 0; %0
c.logitthsa = 0.1000; %2


% Gather prior settings in vectors
c.priormus = [
    c.mu2_0mu,...
    c.logsa2_0mu,...
    c.mu3_0mu,...
    c.logsa3_0mu,...
    c.logitkamu,...
    c.ommu,...
    c.logitthmu,...
             ];

c.priorsas = [
    c.mu2_0sa,...
    c.logsa2_0sa,...
    c.mu3_0sa,...
    c.logsa3_0sa,...
    c.logitkasa,...
    c.omsa,...
    c.logitthsa,...
             ];

% Model function handle
c.prc_fun = @tapas_hgf_categorical;

% Handle to function that transforms perceptual parameters to their native space
% from the space they are estimated in
c.transp_prc_fun = @tapas_hgf_categorical_transp;

return;