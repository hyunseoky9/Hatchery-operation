classdef env2_1 < handle
    properties
        % Basic properties
        envID
        partial
        episodic
        absorbing_cut
        discset
        
        % Discretized spaces
        states
        observations
        actions
        
        % Dimensions of each space
        statespace_dim
        actionspace_dim
        obsspace_dim
        
        % Parameter set index
        parset
        
        % Parameters read from CSV
        alpha0
        alpha1
        sigs
        mu
        kappa
        beta
        muq
        sigq
        NHbar
        Nth
        l
        p
        c
        gamma
        extpenalty
        theta
        sigy
        
        % Current state and observation
        state
        obs
    end
    
    methods
        function obj = env2_0(initstate, parameterization_set, discretization_set)
            % Constructor for Env2_0, translating the Python __init__ method
            
            % Basic setup
            obj.envID = 'Env2.0';
            obj.partial = true;
            obj.episodic = false;
            obj.absorbing_cut = true;  % has an absorbing state, end episode after reaching it
            obj.discset = discretization_set;
            
            % Define state space / observation space / action space
            switch discretization_set
                case 0
                    % States
                    states.NW   = [1000, 3000, 15195, 76961, 389806, 1974350, 10000000];
                    states.NWm1 = [3000, 15195, 76961, 389806, 1974350, 10000000];
                    states.NH   = [0, 75000, 150000, 225000, 300000];
                    states.H    = [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86];
                    states.q    = [65, 322, 457, 592, 848];
                    states.tau  = [0, 1];
                    
                    % Observations
                    observations.y    = [0, 15, 30, 45];  % -1 in Python means spring, here we keep a custom representation
                    observations.ONH  = [0, 75000, 150000, 225000, 300000];
                    observations.OH   = [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86];
                    observations.Oq   = [65, 322, 457, 592, 848];
                    observations.Otau = [0, 1];
                    
                    % Actions
                    actions.a = [0, 75000, 150000, 225000, 300000];
                    
                case 1
                    % States
                    states.NW   = [1000, 2500000, 5000000];
                    states.NWm1 = [2500000, 5000000];
                    states.NH   = [0, 100000, 200000];
                    states.H    = [0.56, 0.71, 0.86];
                    states.q    = [65, 322, 457];
                    states.tau  = [0, 1];
                    
                    % Observations
                    observations.y    = [-1, 0, 30];
                    observations.ONH  = [0, 100000, 200000];
                    observations.OH   = [0.56, 0.71, 0.86];
                    observations.Oq   = [65, 322, 457];
                    observations.Otau = [0, 1];
                    
                    % Actions
                    actions.a = [0, 100000, 200000];
                    
                case 2
                    % States
                    states.NW   = [1000, 3000, 15195, 76961, 389806, 1974350, 10000000];
                    states.NWm1 = [3000, 15195, 76961, 389806, 1974350, 10000000];
                    states.NH   = [0, 75000, 150000, 225000, 300000];
                    states.H    = [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86];
                    states.q    = [65, 322, 457, 592, 848];
                    states.tau  = [0, 1];
                    
                    % Observations
                    observations.y    = [0,2,4,6,8,10,12,14,16,18,20,22,24, ...
                                         26,28,30,32,34,36,38,40,42,44,46];
                    observations.ONH  = [0, 75000, 150000, 225000, 300000];
                    observations.OH   = [0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86];
                    observations.Oq   = [65, 322, 457, 592, 848];
                    observations.Otau = [0, 1];
                    
                    % Actions
                    actions.a = [0, 75000, 150000, 225000, 300000];
                    
                otherwise
                    error('Invalid discretization_set value');
            end
            
            % Assign these structures to the class
            obj.states = states;
            obj.observations = observations;
            obj.actions = actions;
            
            % Compute the dimensionalities
            % Each field in the struct is a vector; we get its length.
            obj.statespace_dim = cellfun(@(x) length(x), struct2cell(obj.states));
            obj.actionspace_dim = cellfun(@(x) length(x), struct2cell(obj.actions));
            obj.obsspace_dim   = cellfun(@(x) length(x), struct2cell(obj.observations));
            
            % Read parameter set from CSV
            parameterization_set_filename = 'parameterization_env2.0.csv';
            paramdf = readtable(parameterization_set_filename);
            
            % Note: Python uses zero-based indexing: paramdf(...) [parameterization_set - 1].
            % In MATLAB, indexing is 1-based. 
            % If your CSV is set up so row 1 corresponds to parameter_set=1, then do:
            rowIndex = parameterization_set;  % direct 1-based indexing
            % If your CSV was aligned differently, adjust accordingly.
            
            % Extract parameter values
            obj.parset      = parameterization_set; % if you still want to store (set - 1)
            obj.alpha0      = paramdf.alpha0(rowIndex);
            obj.alpha1      = paramdf.alpha1(rowIndex);
            obj.sigs        = paramdf.sigs(rowIndex);
            obj.mu          = paramdf.mu(rowIndex);
            obj.kappa       = paramdf.kappa(rowIndex);
            obj.beta        = paramdf.beta(rowIndex);
            obj.muq         = paramdf.muq(rowIndex);
            obj.sigq        = paramdf.sigq(rowIndex);
            obj.NHbar       = paramdf.NHbar(rowIndex);
            obj.Nth         = paramdf.Nth(rowIndex);
            obj.l           = paramdf.l(rowIndex);
            obj.p           = paramdf.p(rowIndex);
            obj.c           = paramdf.c(rowIndex);
            obj.gamma       = paramdf.gamma(rowIndex);
            obj.extpenalty  = paramdf.extpenalty(rowIndex);
            obj.theta       = paramdf.theta(rowIndex);
            obj.sigy        = paramdf.sigy(rowIndex);
            
            % Initialize state and observation
            % (Assumes you have a reset method in this class that takes 'initstate')
            [obj.state, obj.obs] = obj.reset(initstate);
        end

        function [state, obs] = reset(obj, initstate)
            % Initialize empty arrays (row vectors)
            new_state = [];
            new_obs   = [];
        
            % 1. Determine the season
            if initstate(6) == -1
                season = randi([1 2], 1) ;
            else
                season = initstate(6);
            end
            % 2. If spring (season=1)
            if season == 1
                % In Python: new_obs.append(0) => "no observed catch in spring"
                new_obs(end+1) = 1;
                % a) NW index
                if initstate(1) == -1
                    % skip the smallest population size => pick among [2..length(obj.states.NW)]
                    new_state(end+1) = randi([2, length(obj.states.NW)]);
                else
                    new_state(end+1) = initstate(1);
                end
                % b) NWm1 index
                if initstate(2) == -1
                    new_state(end+1) = randi([1, length(obj.states.NWm1)]);
                else
                    new_state(end+1) = initstate(2);
                end
                % c) NH index
                if initstate(3) == -1
                    new_state(end+1) = 1;  % i.e., the first element in obj.states.NH
                    new_obs(end+1)   = 1;
                else
                    new_state(end+1) = initstate(3);
                    new_obs(end+1)   = initstate(3);
                end
                % d) H index
                if initstate(4) == -1
                    idx3 = randi([1, length(obj.states.H)]);
                    new_state(end+1) = idx3;
                    new_obs(end+1)   = idx3;
                else
                    new_state(end+1) = initstate(4);
                    new_obs(end+1)   = initstate(4);
                end
                % e) q index
                if initstate(5) == -1
                    idx4 = randi([1, length(obj.states.q)]);
                    new_state(end+1) = idx4;
                    new_obs(end+1)   = idx4;
                else
                    new_state(end+1) = initstate(5);
                    new_obs(end+1)   = initstate(5);
                end
            % 3. If fall (season=1)
            else
                % a) NW index (skip smallest pop. size if initstate(1) == -1)
                if initstate(1) == -1
                    new_state(end+1) = randi([2, length(obj.states.NW)]);
                else
                    new_state(end+1) = initstate(1);
                end
                % b) Perform fall monitoring
                pop_size  = obj.states.NW(new_state(1));           % actual population size
                new_y_val = obj.fallmonitoring(pop_size);          % call your method
                new_y_cat = obj.discretize(new_y_val, obj.observations.y); 
                % find the index in obj.observations.y that equals new_y_cat
                y_index   = find(obj.observations.y == new_y_cat, 1);
                new_obs(end+1) = y_index;
                % c) NWm1
                if initstate(2) == -1
                    new_state(end+1) = randi([1, length(obj.states.NWm1)]);
                else
                    new_state(end+1) = initstate(2);
                end
                % d) NH
                if initstate(3) == -1
                    idx2 = randi([1, length(obj.states.NH)]);
                    new_state(end+1) = idx2;
                    new_obs(end+1)   = idx2;
                else
                    new_state(end+1) = initstate(3);
                    new_obs(end+1)   = initstate(3);
                end
                % e) H
                if initstate(4) == -1
                    idx3 = randi([1, length(obj.states.H)]);
                    new_state(end+1) = idx3;
                    new_obs(end+1)   = idx3;
                else
                    new_state(end+1) = initstate(4);
                    new_obs(end+1)   = initstate(4);
                end
                % f) q (flow) => "not relevant in fall"
                if initstate(5) == -1
                    new_state(end+1) = 1;  
                    new_obs(end+1)   = 1;
                else
                    new_state(end+1) = initstate(5);
                    new_obs(end+1)   = initstate(5);
                end
            end
            % 4. Append the season 
            new_state(end+1) = season;
            new_obs(end+1)   = season;
            % Store them in object
            obj.state = new_state;
            obj.obs   = new_obs;
            % Return them
            state = new_state;
            obs   = new_obs;
        end


        function [reward, done] = step(obj, action)
            currentNWIdx   = obj.state(1);
            currentNWm1Idx = obj.state(2);
            currentNHIdx   = obj.state(3);
            currentHIdx    = obj.state(4);
            currentqIdx    = obj.state(5);
            currentTauIdx  = obj.state(6);
            
            % Retrieve the actual numeric values from your discretized arrays
            NW   = obj.states.NW(currentNWIdx);
            NWm1 = obj.states.NWm1(currentNWm1Idx);
            NH   = obj.states.NH(currentNHIdx);
            H    = obj.states.H(currentHIdx);
            q    = obj.states.q(currentqIdx);
            tau  = obj.states.tau(currentTauIdx);  % 0 for spring, 1 for fall
        
            % Retrieve the chosen hatchery action
            aVal = obj.actions.a(action);
        
            % but you might want to set it in special cases)
            done = false;
        
            % 2. Check for extinction threshold
            if NW > obj.Nth  
                % 2a. If the population is above threshold => apply normal dynamics
                
                % Next season (tau_next) is 1 - tau
                tau_next = 1 - tau;
        
                % ========== CASE 1: If currently Spring (tau == 0) ==========
                if tau == 0 
                    % 1) Recruitment
                    F = obj.recruitment_rate(q);   % call your custom method
                    H_nextVal = obj.nextgen_heterozygosity(H, NW, NWm1);
                    s = obj.survival_rate(H_nextVal);  % survival depends on next generation heterozygosity
                    recruitment = F * NW;
                    
                    % We'll try binomial( poisson(recruitment), s ) in MATLAB:
                    % If the Poisson call fails for some reason (very large number),
                    % you might revert to normal, as your Python code does.
                    % However, MATLAB doesn't "raise ValueError" the same way,
                    % so we'll just do a try/catch pattern to mimic your logic:
                    try
                        % Poisson draw
                        recruitment_nosurvival = poissrnd(recruitment);  %#ok<*PFBNS> 
                    catch
                        % If something goes wrong with the large Poisson, fallback to normal
                        muNormal = recruitment;
                        sigmaNormal = sqrt(recruitment);
                        recruitment_nosurvival = round(normrnd(muNormal, sigmaNormal));
                        if recruitment_nosurvival < 0
                            recruitment_nosurvival = 0;  % no negative draws
                        end
                    end
                    % binomial draw for survival of recruitment
                    NW_nextVal = obj.sampleBinomial(recruitment_nosurvival, s);

                    
                    % 2) NWm1_nextVal
                    NWm1_nextVal = NW;  % shift the old NW to NWm1
                    % 3) q_nextVal => we can set to 0 or random. Your Python code sets q=0 in spring->fall
                    q_nextVal = 0;
                    % 4) NH_nextVal => action
                    NH_nextVal = action;
                    % 5) Reward => if aVal>0 => p-c else p
                    if aVal > 0
                        reward = obj.p - obj.c;
                    else
                        reward = obj.p;
                    end
        
                % ========== CASE 2: If currently Fall (tau == 1) ==========
                else
                    % (i.e., we are transitioning from Fall to Spring)
                    H_nextVal = obj.update_heterozygosity(H, NW, aVal);
        
                    % If the action is less than or equal to the current NH, use binomial
                    %  -> but note that in your Python code, you do "if a <= NH" => s=...
                    %  -> that means if we put "more fish" than we currently had for NH, it's "invalid".
                    if aVal <= NH
                        s = obj.survival_rate(H_nextVal);
                        NW_nextVal = obj.sampleBinomial(NW + aVal, s);
                        invalidcost = 0;  % no penalty for valid action
                    else
                        % invalid action => NW_next=0, s=0
                        NW_nextVal = 0;
                        s = 0;
                        invalidcost = 0;  % your Python code sets 0 here
                    end
                    % NWm1_nextVal is not updated in fall->spring
                    NWm1_nextVal = NWm1;
                    % q_nextVal => random normal truncated at 0
                    qDraw = normrnd(obj.muq, obj.sigq);
                    if qDraw < 0
                        qDraw = 0;
                    end
                    q_nextVal = qDraw;
        
                    % All hatchery fish that aren't used are discarded => NH=0
                    NH_nextVal = 1;
                    
                    % Reward => p + invalidcost
                    reward = obj.p + invalidcost;
                end
        
                % 2b. Discretize NW_nextVal, H_nextVal, q_nextVal
                NW_nextVal = obj.discretize(NW_nextVal, obj.states.NW);
                H_nextVal  = obj.discretize(H_nextVal,  obj.states.H);
                q_nextVal  = obj.discretize(q_nextVal,  obj.states.q);
        
                % 2c. Observation for the next season
                if tau == 0 
                    y_nextVal = obj.fallmonitoring(NW_nextVal);
                    y_nextVal = obj.discretize(y_nextVal, obj.observations.y);
                else
                    % next season is Spring => no observed catch
                    y_nextVal = 0; 
                end
        
                % 2d. Convert to discrete *indices* for the new state
                NW_next_idx = find(obj.states.NW == NW_nextVal, 1);
                NWm1_next_idx = find(obj.states.NWm1 == NWm1_nextVal, 1);
                H_next_idx = find(obj.states.H == H_nextVal, 1);
                q_next_idx = find(obj.states.q == q_nextVal, 1);
                tau_next_idx = find(obj.states.tau == tau_next, 1);
                % For the observation's y, find the index in obj.observations.y
                y_next_idx = find(obj.observations.y == y_nextVal, 1);
        
                % Construct the new state (6D)
                nextState = [ ...
                    NW_next_idx, ...
                    NWm1_next_idx, ...
                    NH_nextVal, ...   % <-- NH_nextVal was an integer from "action" or 0, no indexing needed
                    H_next_idx, ...
                    q_next_idx, ...
                    tau_next_idx ...
                ];
        
                % Construct the new observation (5D)
                nextObs = [ ...
                    NW_next_idx, ...
                    NWm1_next_idx, ...
                    NH_nextVal, ...   % <-- NH_nextVal was an integer from "action" or 0, no indexing needed
                    H_next_idx, ...
                    q_next_idx, ...
                    tau_next_idx ...
                ];
                done = false;
            else
                % 3. If the population is below or equal to Nth => "extinct"
                tau_next = 1 - tau;
                
                if tau == 0
                    % If current season is Spring => next is Fall
                    % q_nextVal => can set to the first element or random normal.
                    % In your code you do: q_next = self.states['q'][0].
                    % Let's replicate that exactly: q_nextVal = obj.states.q(1).
                    q_nextVal = obj.states.q(1);
        
                    % If aVal>0 => extpenalty - c, else extpenalty
                    if aVal > 0
                        reward = obj.extpenalty - obj.c;
                    else
                        reward = obj.extpenalty;
                    end
                    
                    NH_nextVal = action;  % as in your code
                    y_nextVal  = 0;     % next observation (Fall) => your code sets it to 0 in this path.
                    
                else
                    % If current season is Fall => next is Spring
                    % q_nextVal => random normal truncated at 0
                    qDraw = normrnd(obj.muq, obj.sigq);
                    if qDraw < 0
                        qDraw = 0;
                    end
                    % discretize
                    q_nextVal = obj.discretize(qDraw, obj.states.q);
        
                    NH_nextVal = 1;  
                    y_nextVal  = 0;  % no observed catch in spring
                    reward = obj.extpenalty;
                end
        
                % If tau == 0, we do a special check for "next season is fall" => y_nextVal=0
                % (Your Python code sets y_next=0 anyway. Already done above.)
        
                % Next states are all set to 0 except q
                NW_next_idx   = 1;  % index 1 => the smallest NW in your array (like "0" in python)
                NWm1_next_idx = 1;
                H_next_idx    = 1;
                tau_next_idx = find(obj.states.tau == tau_next, 1);
        
                % We have q_nextVal as a numeric value. Let's find which discrete index it is:
                q_next_idx = find(obj.states.q == q_nextVal, 1);
        
                % For y, find the matching index
                y_next_idx = find(obj.observations.y == y_nextVal, 1);
        
                nextState = [ NW_next_idx, NWm1_next_idx, NH_nextVal, H_next_idx, q_next_idx, tau_next_idx ];
                nextObs   = [ NW_next_idx, NWm1_next_idx, NH_nextVal, H_next_idx, q_next_idx, tau_next_idx ];
        
                % done remains false in your code
            end
            
            % 4. Store them in the object for reference
            obj.state = nextState;
            obj.obs   = nextObs;
        
            % ---------------------------------------------------
            % 5. Return them
            % ---------------------------------------------------
            % Typically, RL step methods return (nextState, reward, done, info)
            % but you might want to return nextObs as well, depending on your design.
            %
            % This function returns them in the signature:
            %   [nextState, nextObs, reward, done]
        end

        function val = fallmonitoring(obj, NW)
            % Fallback monitoring: Negative binomial with parameters (r, p)
            % where r = obj.sigy, p = obj.sigy/(expcatch + obj.sigy)
            expCatch = obj.theta * NW;
            
            r = obj.sigy;                            % "number of successes" in NB terms
            p = obj.sigy / (expCatch + obj.sigy);    % success probability in NB
            
            % nbinrnd(r,p) yields the number of failures before r successes.
            % This matches numpy.random.negative_binomial(r,p) if we interpret
            % the parameterization consistently (expected value = r*(1-p)/p).
            val = nbinrnd(r, p);
        end

        function s = survival_rate(obj, H)
            % Adds Normal(0, obj.sigs) noise, then exponentiates the negative
            epsVal = randn() * obj.sigs;  % randn is standard normal in MATLAB
            val = obj.alpha0 + obj.alpha1 * (1 - H) + epsVal;
            s = exp(-max(val, 0));
        end

        function F = recruitment_rate(obj, q)
            F = obj.beta * q;
        end

        function Hnext = update_heterozygosity(obj, H, NW, a1)
            % Weighted average of hatchery and wild heterozygosity
            Hnext = (a1 * H * (1 - obj.l) + NW * H) / (a1 + NW);
        end

        function Hnext = nextgen_heterozygosity(obj, H, NW, NWm1)
            % Effective population size
            Ne = 2.0 ./ (1.0./NW + 1.0./NWm1);
            
            % Mode of Beta distribution
            modeVal = H * (1.0 + (obj.mu - 1.0/(2.0 * Ne)));
            
            % Beta parameters
            aParam = modeVal * (obj.kappa - 2.0) + 1.0;
            bParam = (1.0 - modeVal) * (obj.kappa - 2.0) + 1.0;
            
            % Sample from Beta distribution
            Hnext = betarnd(aParam, bParam);
        end

        function val = discretize(obj, x, possibleStates)
            % DISCRETIZE finds the two closest bin values in possibleStates
            % to x, then randomly chooses one with probability proportional
            % to the inverse distance.
            %
            % Inputs:
            %   x              : the continuous (or integer) value to discretize
            %   possibleStates : a sorted vector of discrete bins
            %
            % Output:
            %   val : the chosen bin from possibleStates
            
            % 1) Clamp if x < min or x > max
            if x <= possibleStates(1)
                val = possibleStates(1);
                return;
            elseif x >= possibleStates(end)
                val = possibleStates(end);
                return;
            end
            
            % 2) Otherwise, find the two closest bins to x
            diffs = abs(possibleStates - x);
            [~, sortIdx] = sort(diffs, 'ascend');
            
            % sortIdx(1) is the index of the closest bin
            % sortIdx(2) is the index of the second closest bin
            lowerVal = possibleStates(sortIdx(1));
            upperVal = possibleStates(sortIdx(2));
            
            % Ensure lowerVal <= upperVal
            if lowerVal > upperVal
                temp = lowerVal; 
                lowerVal = upperVal; 
                upperVal = temp;
            end
            
            % 3) Compute weights based on distance
            distRange = upperVal - lowerVal;
            if distRange == 0
                % x is exactly on a discrete bin, or the two bins are the same
                val = lowerVal;
                return;
            end
            
            wLower = (upperVal - x) / distRange;  % Probability of picking the lower bin
            
            % 4) Randomly choose lowerVal or upperVal based on these weights
            r = rand();
            if r < wLower
                val = lowerVal;
            else
                val = upperVal;
            end
        end

        function NW_nextVal = sampleBinomial(obj, x, s)
            % sampling binomial distribution with normal dist. approximation if n is too large.
            LARGE_THRESHOLD = 1e4;
        
            if x < LARGE_THRESHOLD
                NW_nextVal = binornd(x, s);
        
            else
                % Use a normal approximation for large x
                mu  = x * s;                  % mean
                var = x * s * (1 - s);        % variance
                sig = sqrt(var);
        
                % Sample from Normal; round to integer
                NW_nextVal = round(normrnd(mu, sig));
        
                % Clip to 0 if negative
                if NW_nextVal < 0
                    NW_nextVal = 0;
                end
            end
        end

    end
end