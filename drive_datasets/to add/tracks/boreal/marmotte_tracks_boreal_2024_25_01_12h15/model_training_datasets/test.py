import pandas as pd 
import numpy as np 

path_to_input_space = "drive_datasets/data/marmotte/tracks/tile/marmotte_tracks_tiles_2023_12_12_12h15/model_training_datasets/input_space_data.pkl"
print(pd.read_pickle(path_to_input_space).columns)

df = pd.read_pickle(path_to_input_space)

[print(col,df[col]) for col in df.columns]


# marmotte
r= 0.12
wheel_baseline= 0.52

j = np.array([[1/2, 1/2], [-1/wheel_baseline, 1/wheel_baseline]])

command_space_body = np.hstack(df["maximum_wheel_vel_positive [rad/s]"])

#df["maximum_wheel_vel_positive [rad/s]"] = 12#df[]
#df["maximum_wheel_vel_negative [rad/s]"] = 12#-3.5

#df.to_pickle(path_to_input_space)




