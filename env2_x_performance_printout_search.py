import os

wd = './DRQN results/hpc outputs/save3'
# get files
files = os.listdir(wd)

for file in files:
    if file.endswith('.out'):
        # read file into a list 
        with open(wd + '/' + file, 'r') as f:
            lines = f.readlines()
        # receptacles
        performance = []
        seedperformance = None
        i=0
        for line in lines:
            # if the line says management was done, save it to a list
            if 'Episode' in line:
                strsplit = line.split(': ')
                strsplit2 = line.split(' ')
                score = float(strsplit[-1].strip())
                seed = int(strsplit2[3][:-1])
                if score > 4000.0:
                    performance.append((seed,score))
            elif 'final average reward' in line:
                strsplit = line.split(': ')
                strsplit2 = line.split(' ')
                score = float(strsplit[-1].strip())
                seed = int(strsplit2[4][:-2])
                if score > 4000.0:
                    performance.append((seed,score))
            i += 1

        print(f'file: {file}')
        # print the seed number and the performance
        if len(performance) > 0:
            for i in range(len(performance)):
                print(f'performance: {performance[i][1]}, seed: {performance[i][0]}')
            print('\n')
        
