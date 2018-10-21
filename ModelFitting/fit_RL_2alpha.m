function params = fit_RL_2alpha(rew, vol, choices)
% Function to find the set of reinforcement learning parameters that
% minimize the negative log likelihood. For model with two learning rates
%
%INPUT:
%   -rew: task reward structure
%   -vol:Volatility condition (1 = high volatility, 2 = low volatility)
%   -choices: participant choices
%
%OUTPUT:
%   - params: best fitting parameters obtained by the solver

%% INITIAL PARAMTERS
% Set upper and lower parameter bounds - the algorithm will only search in
% this parameter space
ub = [1 1 100];
lb = [0 0 0];

% Set starting parameters (where the algorithm should start)
ps0 = [.5 .5 2];

%% FITTING
model = @fitModel; %fevalable using the loss function defined below

opts = optimoptions(@fmincon, 'Algorithm','interior-point','Display','off'); %options for the multistart function
ms = MultiStart('UseParallel','always'); %initialize mutlistart solver
problem = createOptimProblem('fmincon','objective',model,'x0',ps0,'lb',lb,'ub',ub,'options',opts);
params = run(ms,problem,16); %the 16 here specifies the number of solvers to run in parallel

%% LOSS FUNCTION
    function loss = fitModel(ps)
        loss = RL_2alpha_LL(rew,vol,ps(1),ps(2),ps(3),choices); %here I'm using negative log-likelihood as my loss function
    end
end