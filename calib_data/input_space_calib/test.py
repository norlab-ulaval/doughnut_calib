import pandas as pd 




df = pd.read_pickle("DRIVE/calib_data/input_space_calib/husky_input_space_data.pkl")

print(df)


df["maximum_wheel_vel_positive [rad/s]"] = 6.0569351907934585
df["maximum_wheel_vel_negative [rad/s]"] = -6.0569351907934585
print()