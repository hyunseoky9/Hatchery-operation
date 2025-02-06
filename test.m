initstate           = [-1, -1, -1, -1, -1, -1];  % randomize all
parameterizationSet = 1;
discretizationSet   = 0;
env = env2_0(initstate, parameterizationSet, discretizationSet);

% 2. (Optional) reset again to ensure we start fresh
[state, obs] = env.reset([-1, -1, -1, -1, -1, -1]);
fprintf("Initial state: "); disp(state)
fprintf("Initial obs: ");   disp(obs)

% 3. Step a few times
for t = 1:5
    % Let's pick an action at random from the available actions
    numActions = length(env.actions.a);  % how many discrete actions
    actionIndex = randi(numActions);     % random int in [1..numActions]
    
    [nextState, nextObs, reward, done] = env.step(actionIndex);
    
    fprintf("t=%d, Action=%d => NextState=", t, actionIndex);
    disp(nextState);
    fprintf("NextObs=");
    disp(nextObs);
    fprintf("Reward=%g, done=%d\n\n", reward, done);
    
    % If done is true, break out
    if done
        break;
    end
end