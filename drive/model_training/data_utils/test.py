import numpy as np 
import pandas as pd
from extractors import *
import matplotlib.pyplot as plt

test = [[10,10],[20,20]]

flattened_list = [item for sublist in test for item in sublist]
print(flattened_list)