function data = RL_2alpha(rew,vol,alpha1,alpha2,beta)
%% FUNCTION TO GENERATE RESPONSES FROM A REINFORECMENT LEARNING MODEL
% Function to model the behavior of a reinforcement learning agent alphas
% that depend on the environments volatility.
%
% INPUT:
%    - rew: Reward payout matrix for selecting either item on each trial
%    - vol: trial-by-trial index of the current volatility block (1 high
%    volatility, 2 low volatility)
%    - alpha1, alpha2: learning rate parameters (ranges between 0 and 1)
%    - beta: inverse temperature parameter reflecting choice variability -
%    values tending towards 0 indicate random choices, values tending
%    towards infinity indicate hard-max "greedy" options
%
% OUTPUT: structure "data" with following arrays
%    - ParamNames: Parameter names
%    - ParamValues: Values of the paramters used
%    - QValues: Trial-by-trial Q-values for each option
%    - Pchoices: Probability of selecting each option after softmax
%    - Choices: Simulated choices
%    - Delta: Trial-by-trial prediction error
%    - rew: Reward matrix
%    - Reward: Trial-by-trial reward received


%% INITIALIZE ARRAYS
ntrials = length(rew(:,1));  %Number of trials
V = NaN(ntrials,2);          %Q-values at the start of each trial
Vinit = [.5 .5];             %Initial Q-values
Pchoice = NaN(ntrials,2);    %Probability of making each choice on each trial
Choice = ones(ntrials,1);    %Stochastic choice made for a given simulation
Delta = NaN(ntrials,1);      %Reward prediction error
Reward = NaN(ntrials,1);      %Whether or not a reward was obtained

%% RUN MODEL
for i = 1:ntrials
    % Initialize equal Q values on the first trial
    if i == 1
        V(i,:) = Vinit;
    end
    
    % Get probability of making a choice given the current Q-values
    Pchoice(i,:) = exp(beta.*V(i,:))./sum(exp(beta.*V(i,:)));
 
    % Make choice - 1 for left 2 for right
    if rand < Pchoice(i,2)
        Choice(i) = 2;
    end
    
    % Get reward prediction error (delta)
    Reward(i) = rew(i,Choice(i));
    Delta(i) = Reward(i)-V(i,Choice(i));
    
    %Update Q-value for the next trial
    if i < ntrials
        V(i+1,:) = V(i,:);
        if vol(i) == 1
            V(i+1,Choice(i)) = V(i,Choice(i))+(alpha1*Delta(i));
        elseif vol(i) == 2
            V(i+1,Choice(i)) = V(i,Choice(i))+(alpha2*Delta(i));
        end
    end
end

%% SAVE ARRAYS
data.ParamNames = {'Alpha1','Alpha2','Beta'};
data.ParamValues = [alpha1,alpha2,beta];
data.QValues = V;
data.Pchoice = Pchoice;
data.Choices = Choice;
data.Delta = Delta;
data.rew = rew;
data.Reward=Reward;
end
    