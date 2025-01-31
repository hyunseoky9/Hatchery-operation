import sys
import os
import numpy
import math
import random
import shutil
import subprocess
import pickle
import torch
if len(sys.argv) != 2:
  print('Usage: %s MAXIMUM' % (os.path.basename(sys.argv[0])))
  sys.exit(1)

maximum = int(sys.argv[1])

n1 = 1
n2 = 1

while n2 <= maximum:
  n1, n2 = n2, n1 + n2
print('torch is loaded!')
print('The greatest Fibonacci number up to %d is %d' % (maximum, n1))
