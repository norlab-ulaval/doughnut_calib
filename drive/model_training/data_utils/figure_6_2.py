import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

NBR_STEPS = 120
TIME_DELTA = 0.05
PATH_DATAFRAME = "drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_max_lin_speed_all_speed_all_terrain_steady_state_dataset.pkl"

class Command:
    def __init__(self, cmd_vel, cmd_angle, delta_s, step_nb):
        self.cmd_vel = cmd_vel
        self.cmd_angle = cmd_angle
        self.delta_s = delta_s
        self.step_nb = step_nb
        delta_x = cmd_vel*delta_s
        delta_z_dot = cmd_angle*delta_s
        self.cmd_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.cmd_matrix.astype(float)
        self.cmd_matrix[0, 0] = np.cos(delta_z_dot)
        self.cmd_matrix[0, 1] = -np.sin(delta_z_dot)
        self.cmd_matrix[1, 0] = np.sin(delta_z_dot)
        self.cmd_matrix[1, 1] = np.cos(delta_z_dot)
        self.cmd_matrix[0, 2] = delta_x


def process_path(cmd : Command):
    planned_path = []
    matrice_pose_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range (cmd.step_nb):
        matrice_pose_init = matrice_pose_init @ cmd.cmd_matrix
        theta_z = np.arctan2(matrice_pose_init[1, 0], matrice_pose_init[0, 0])
        rot = R.from_euler('zyx', [theta_z, 0.0, 0.0])
        quat = rot.as_quat()
        # [x, y, yaw]
        planned_path.append([matrice_pose_init[0, 2], matrice_pose_init[1, 2], quat[2]])
    return planned_path


def draw_path(ax, cmd, limits):
    planned_path = process_path(cmd)
    planned_path = np.array(planned_path)
    ax.plot(-planned_path[:, 1], planned_path[:, 0])
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title("Planned path")
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.grid()


def plot_dataframe(df):
    # Find every terrain in the dataframe
    terrains = df['terrain'].unique()
    for terrain in terrains:
        # Filter the dataframe to keep only the rows corresponding to the current terrain
        df_terrain = df[df['terrain'] == terrain]
        # Read the number of rows in the dataframe to get the number of commands
        cmd_nbr = df_terrain.shape[0]
        # Create a square grid of subplots to display the path of the robot for each command
        row_nbr = int(np.ceil(np.sqrt(cmd_nbr)))
        col_nbr = int(np.ceil(np.sqrt(cmd_nbr)))
        fig, axs = plt.subplots(row_nbr, col_nbr, figsize=(20, 20))
        list_cmd_vel = df_terrain['cmd_body_x_lwmean'].values
        list_cmd_angle = df_terrain['cmd_body_yaw_lwmean'].values
        list_max_linear_speed = df_terrain['max_linear_speed_sampled'].values
        # Loop over the rows of the dataframe
        for cmd_vel, cmd_angle, max_linear_speed, i in zip(list_cmd_vel, list_cmd_angle, list_max_linear_speed, range(cmd_nbr)):
            # Get the command from the dataframe
            cmd = Command(cmd_vel, cmd_angle, TIME_DELTA, NBR_STEPS)
            range_limit = max_linear_speed * NBR_STEPS * TIME_DELTA
            # Draw the path of the robot
            draw_path(axs[i//col_nbr, i%col_nbr], cmd, range_limit)

        # Increase the spacing between the subplots
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"tests_figures/planned_path_{terrain}.png")


### Test the functions
cmd_vel = 5.0
cmd_angle = 0.2
delta_s = 0.05
step_nb = 120
cmd = Command(cmd_vel, cmd_angle, delta_s, step_nb)
fig, ax = plt.subplots()    
draw_path(ax, cmd, 30)
plt.savefig("tests_figures/planned_path_single.png")

# Load the dataframe
df_all_terrain = pd.read_pickle(PATH_DATAFRAME)
# Plot the path of the robot for each command in the dataframe
plot_dataframe(df_all_terrain)
plt.savefig("tests_figures/planned_path_multiple.png")
