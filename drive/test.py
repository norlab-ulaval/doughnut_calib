import numpy as np 
import pandas as pd
new_row = np.array([1,2,3,4])

new_row = np.hstack((new_row,np.array([10,20],dtype=np.float64)))



df = pd.read_pickle("/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/raw_dataframe_wronged_calib_step.pkl")
print(df.columns)
print(len(df["meas_right_vel"].unique()))
print(len(df["meas_left_vel"].unique()))

df = pd.read_pickle("/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/raw_dataframe.pkl")
print(df.columns)
print(len(df["meas_right_vel"].unique()))
print(len(df["meas_left_vel"].unique()))