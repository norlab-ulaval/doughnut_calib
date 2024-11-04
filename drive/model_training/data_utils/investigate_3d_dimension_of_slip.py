import pandas as pd

import sys
print(sys.version)
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../../')


from drive.model_training.data_utils.extractors import * 
from drive.model_training.data_utils.animate_time_constant import * 
from first_order_model import *

from drive.util.model_func import *
from drive.util.transform_algebra import *



import matplotlib.animation as animation
from matplotlib.backend_bases import KeyEvent

import tqdm
import pickle

from matplotlib.backends.backend_pdf import PdfPages

color_dict = {"asphalt":"grey", "ice":"blue","gravel":"orange","grass":"green"}

steady_state_path = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/results_multiple_terrain_dataframe/all_terrain_steady_state_dataset.pkl"
df_diamond = pd.read_pickle(steady_state_path)

df_warthog = df_diamond.loc[df_diamond["robot"]=="warthog"]

df_sampling_speed = df_warthog.loc[df_warthog["max_linear_speed_sampled"]==5.0]

print_column_unique_column(df_sampling_speed)

list_slip_columns = ['slip_body_x_ss','slip_body_y_ss','slip_body_yaw_ss']
# Compute direction slip 

slip = 'slip_body_norm_x_y_ss'
df_sampling_speed[slip] = np.sqrt(df_sampling_speed.slip_body_x_ss**2 + df_sampling_speed.slip_body_y_ss**2) 


slip = 'slip_body_y_ss'

# Extract terrain 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


df_terrain = df_sampling_speed.loc[df_sampling_speed.terrain == "grass"]
ax.scatter(df_terrain.slip_body_y_ss,df_terrain.cmd_body_x_lwmean, df_terrain[slip], c=color_dict[terrain], marker='o')


plt.show()

list_terrain_1 = ["ice"]
list_terrain = df_sampling_speed.terrain.unique()

for terrain in list_terrain:

    df_terrain = df_sampling_speed.loc[df_sampling_speed.terrain == terrain]
    
    # Scatter plot
    ax.scatter(df_terrain.cmd_body_yaw_lwmean,df_terrain.cmd_body_x_lwmean, df_terrain[slip], c=color_dict[terrain], marker='o')

# Labels and title
ax.legend()
ax.set_xlabel('cmd_body_yaw')
ax.set_ylabel('cmd_body_x')
ax.set_zlabel(slip)
ax.set_title('3D Scatter Plot')
ax.set_zlim(-3,3)
# Show the plot
plt.show()


###

