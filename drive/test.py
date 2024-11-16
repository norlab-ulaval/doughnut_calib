import numpy as np 
import pandas as pd
new_row = np.array([1,2,3,4])

new_row = np.hstack((new_row,np.array([10,20],dtype=np.float64)))

df = pd.read_pickle("drive_datasets/data/warthog/wheels/gravel/warthog_wheels_gravel_2024_8_6_11h37s32/model_training_datasets/raw_dataframe.pkl")


print(df[["calib_step"]])

print(1727290198.655643344 - 1727290198.655643344)

