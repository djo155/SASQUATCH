function NLL = RL_2alpha_LL(rew,vol,alpha1,alpha2,beta,choices)

%% INITIALIZE ARRAYS
ntrials = length(rew(:,1));  %Number of trials
V = NaN(ntrials,2);          %Q-values at the start of each trial
Vinit = [.5 .5];             %Initial Q-values
Pchoice = NaN(ntrials,2);    %Probability of making each choice on each trial
LL = 0; %log likelihood

%% GET CHOICE PROBABILITIES USING PARTICIPANT CHOICES
for i = 1:ntrials
    if i == 1
        V(i,:) = Vinit;
    end
    
    % Get probability of making a choice given the current Q-values
    Pchoice(i,:) = exp(beta.*V(i,:))./sum(exp(beta.*V(i,:)));
 
    % Get reward prediction error (delta)
    reward = rew(i,choices(i));
    delta = reward-V(i,choices(i));
    
    % Get log-likelihood that the model with these parameters would have
    % made the same choice as the subject
    LL = LL+log(Pchoice(i,choices(i)));
    
    %Update Q-value for the next trial
    if i < ntrials
        V(i+1,:) = V(i,:);
        if vol(i) == 1
            V(i+1,choices(i)) = V(i,choices(i))+(alpha1*delta);
        elseif vol(i) == 2
            V(i+1,choices(i)) = V(i,choices(i))+(alpha2*delta);
        end
    end
end

%% SAVE ARRAYS
NLL = -LL; %Save negative log-likelihood for gradient decent
end
    