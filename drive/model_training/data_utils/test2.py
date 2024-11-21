import pandas as pd 

from extractors import *



df = pd.read_pickle("drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_max_lin_speed_5.0_all_terrain_steady_state_dataset.pkl")

print_column_unique_column(df)

#
#"step_frame_icp_x","step_frame_icp_y","step_frame_icp_yaw"

data = column_type_extractor(df,'step_frame_interpolated_icp_x')
data_2 = column_type_extractor(df,'step_frame_interpolated_icp_y')
fig, ax = plt.subplots(1,1)

print(data)
step = 1000
ax.plot( data[step,:],data_2[step,:])

plt.show()