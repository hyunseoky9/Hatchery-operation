import os

wd = './DRQN results/hpc outputs'
# get files
files = os.listdir(wd)

for file in files:
    if file.endswith('.out'):
        # read file into a list 
        with open(wd + '/' + file, 'r') as f:
            lines = f.readlines()
        
