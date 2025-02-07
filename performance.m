%rng(0)
episodenum = 10000;
maxstep = 100;

initstate           = [-1, -1, -1, -1, -1, -1];  % randomize all
parameterizationSet = 2;
discretizationSet   = 0;
env = env2_0(initstate, parameterizationSet, discretizationSet);
numActions = length(env.actions.a);  % how many discrete actions
totrewards = 0;
strat = 1;
for i = 1:episodenum
    fprintf('episode num: %d\n',i)
    % 2. (Optional) reset again to ensure we start fresh
    env.reset([-1, -1, -1, -1, -1, -1]);
    totreward = 0;
    % 3. Step a few times
    for t = 1:maxstep
        % Let's pick an action at random from the available actions
        if strat == 0 % no augmentation
            actionIndex = 1;
        elseif strat == 1 % max augmentation
            if env.state(6) == 1 % spring
                actionIndex = numActions;
            else % Fall
                actionIndex = env.state(3);
            end
        elseif strat == 2 % random augmentation
            actionIndex = randi(numActions);     % random int in [1..numActions]            
        end
        [reward, done] = env.step(actionIndex);
        totreward = totreward + reward;
        % If done is true, break out
        if done
            break;
        end
    end
    totrewards = totrewards + totreward;
end
fprintf("Average reward: %f\n", totrewards/episodenum);