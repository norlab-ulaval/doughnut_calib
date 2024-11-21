import pandas as pd 


df = pd.read_pickle("/home/robot/ros2_ws/src/DRIVE/calib_data/husky_wheels_avide_2024_11_3_12h43s16/model_training_datasets/raw_dataframe.pkl")


print(df.columns)

for col in list(df.columns):

    print(df[col].describe())