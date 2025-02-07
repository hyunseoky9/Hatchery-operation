%rng(0)
initstate           = [-1, -1, -1, -1, -1, -1];  % randomize all
parameterizationSet = 2;
discretizationSet   = 0;
env = env2_0(initstate, parameterizationSet, discretizationSet);


episodenum = 10000;
numsteps = [];
for k = 1:episodenum
    % 2. (Optional) reset again to ensure we start fresh
    [state, obs] = env.reset([-1, -1, -1, -1, -1, -1]);
    % 3. Step a few times
    state_save = env.state;
    obs_save = env.obs;
    extinct_period = -1;
    extinct_recorded = 0;
    for t = 1:100
        % Let's pick an action at random from the available actions
        actionIndex = 1;     % random int in [1..numActions]
        [reward, done] = env.step(actionIndex);
        if env.state(1) == 1 && extinct_recorded == 0
            extinct_period = t;
            numsteps = [numsteps; extinct_period];
            extinct_recorded = 1;
        end
        % save
        state_save = [state_save; env.state];
        obs_save = [obs_save; env.obs];
        % If done is true, break out
        if done
            break;
        end
    end
    if extinct_period == -1
        numsteps = [numsteps; 100];
    end
end
% Create a simple time array
timesteps = 1:size(state_save, 1);

% Reuse the same figure number each time
figure(1);       % direct MATLAB to use (or create) figure #1
clf;             % clear old plots in figure #1, so you start fresh

% Plot NW index vs time
%subplot(2,1,1);
%plot(timesteps, state_save(:,1), 'o-'); 
%xlabel('Time step');
%ylabel('NW index');
%title('NW index across timesteps');
%
%% Plot y index vs time
%subplot(2,1,2);
%plot(timesteps, obs_save(:,1), 'o-');
%xlabel('Time step');
%ylabel('y index');
%title('y index across timesteps');

% histogram of numsteps
histogram(numsteps, 100);
xlabel('Extinction time');
ylabel('Frequency');
title('Histogram of extinction time');
