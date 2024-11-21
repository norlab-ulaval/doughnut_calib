import numpy as np 
import pandas as pd
from extractors import *
import matplotlib.pyplot as plt

df = pd.read_pickle("drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_max_lin_speed_all_all_terrain_steady_state_dataset.pkl")

print_column_unique_column(df)


df.plot("init_tf_pose_x","init_tf_pose_y")
plt.show()