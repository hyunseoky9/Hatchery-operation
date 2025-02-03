import os

wd = './DRQN results/hpc outputs/save1'
# get files
files = os.listdir(wd)

for file in files:
    if file.endswith('.out'):
        # read file into a list 
        with open(wd + '/' + file, 'r') as f:
            lines = f.readlines()
        # receptacles
        seednum = []
        performance = []
        seedperformance = None
        i=0
        for line in lines:
            # if the line says management was done, save it to a list
            if 'seed:' in line:
                # append last seed's performances
                if seedperformance is not None:
                    performance.append(seedperformance)
                seed_number = int(line.split(': ')[1])
                seednum.append(seed_number)
                seedperformance = []
            if 'Episode' in line or 'final average reward' in line:
                score = float(line.split(': ')[-1].strip())
                if score > 4000.0:
                    seedperformance.append(score)
            i += 1
        # append last seed's performances
        performance.append(seedperformance)

        print(f'file: {file}')
        # print the seed number and the performance
        for i in range(len(seednum)):
            if len(performance[i]) > 0:
                print(f'seed number: {seednum[i]}')
                for j in range(len(performance[i])):
                    print(f'performance: {performance[i][j]}')
                print('\n')
        
