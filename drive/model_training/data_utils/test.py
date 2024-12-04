import numpy as np 
import pandas as pd
from extractors import *
import matplotlib.pyplot as plt



body_vel_x = 10
body_vel_yaw = 10 
body_vel_y = 2

maximal_increment = 0.5

n_increment_x = body_vel_x//maximal_increment
n_increment_y = body_vel_y//maximal_increment
n_increment_yaw = body_vel_yaw//maximal_increment

print(n_increment_x, n_increment_y, n_increment_yaw)

