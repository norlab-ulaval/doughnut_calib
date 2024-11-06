import pathlib 
import pandas as pd 
import extractors
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_pickle("launch/test.pkl")

extractors.print_column_unique_column(df)

df["right_wheel_current"] = df["right_wheel_current"].astype(float)
df["left_wheel_current"] = df["left_wheel_current"].astype(float)
df["left_wheel_voltage"] = df["left_wheel_voltage"].astype(float)
df["right_wheel_voltage"] = df["right_wheel_voltage"].astype(float)

# Get descriptive statistics
left_wheel_current_desc = df["left_wheel_current"].describe()
left_wheel_voltage_desc = df["left_wheel_voltage"].describe()
right_wheel_voltage_desc = df["right_wheel_voltage"].describe()

# Print the results
print("Left Wheel Current Description:\n", left_wheel_current_desc)
print("\nLeft Wheel Voltage Description:\n", left_wheel_voltage_desc)
print("\nRight Wheel Voltage Description:\n", right_wheel_voltage_desc)


print(np.sqrt(5000/500/0.5))

