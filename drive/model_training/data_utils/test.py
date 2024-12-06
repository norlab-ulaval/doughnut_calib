import numpy as np 
import pandas as pd
from extractors import *
import matplotlib.pyplot as plt


terrain_shrinkage = np.array([65.62,63.83,62.07,59.65])

print(np.mean(terrain_shrinkage))
print(np.std(terrain_shrinkage))
