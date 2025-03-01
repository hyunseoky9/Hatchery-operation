from  DRQN import *
from call_paramset import call_paramset, call_env

if __name__ == "__main__":
    
    paramid = 37
    iteration_num = 3
    hyperparameterization_set_filename = './hyperparamsets/DRQN_hyperparameters.csv'
    paramdflist = call_paramset(hyperparameterization_set_filename,paramid)
    tuneset = 1
    for paramdf in paramdflist:
        for iteration in range(iteration_num): 
            # process hyperparameter dataframe
            # set seed
            seed = paramdf['seed'] # make sure iteration_num matches with the number of seeds if seeds are specified
            if seed == 'random':
                seednum = random.randint(0,1000000)
            else:
                seednum = int(seed)
            os.environ["PYTHONHASHSEED"] = str(seednum)
            random.seed(seednum)
            np.random.seed(seednum)
            torch.manual_seed(seednum)
            # environment configuration
            env = call_env(paramdf) 
            meta = {'paramid': paramid, 'iteration': iteration, 'seed': seednum}
            rewards, final_avgreward = DRQN(env, paramdf, meta)
        tuneset += 1