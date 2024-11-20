import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

NBR_STEPS = 120
TIME_DELTA = 0.05
PATH_DATAFRAME = "drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_max_lin_speed_all_speed_all_terrain_steady_state_dataset.pkl"
INDEX_LIST_TO_PLOT = [2, 3, 12, 20, 25, 30, 35, 41, 47, 122]
# Create a list of colors to plot the paths of the robot from the inferno colormap of equal length to the index list
COLOR_LIST_TO_PLOT = plt.cm.inferno(np.linspace(0, 1, len(INDEX_LIST_TO_PLOT)))
RANGE_LIMIT = 20

class Command:
    def __init__(self, cmd_vel, cmd_angle, delta_s, step_nb, initial_pose=[0.0, 0.0, 0.0]):
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
        self.initial_pose = np.array(initial_pose)


def extracts_appropriate_columns(df,commun_name):
    df_columns = list(df.columns)
    possible_number = ["1","2","3","4","5","6","7","8","9","0"]

    for i,column in enumerate(df_columns):
        if column[-1] in possible_number:
            if column[-2] in possible_number:
                df_columns[i] = column[:-3]
                if column[-3] in possible_number:
                    df_columns[i] = column[:-4]
            else:
                df_columns[i] = column[:-2]

    df_columns_name = pd.Series(df_columns)
    mask = [name == commun_name for name in df_columns_name ]
    kept_column = df.columns[mask]

    return df[kept_column]


def column_type_extractor(df, common_type,
                        transient_state=True,steady_state=True, verbose=False):
    """ Extract the np.matrix that represent the 40 columns of all steps of the 
    specific type. 
    column_type_extractor
    For example, icp_velx_40 is of the type icp_velx 
    
    """
    local_df = df.copy()
    if transient_state==False and steady_state==True:
        mask = local_df.steady_state_mask == 1
        local_df = local_df.loc[mask]
    elif steady_state == False and transient_state == True:
        mask = local_df.steady_state_mask == 0
        local_df = local_df.loc[mask]
    elif steady_state == False and transient_state == False:
        raise ValueError("Both steady state and transient can not be at false")
    
    local_df = extracts_appropriate_columns(pd.DataFrame(local_df),common_type)
    np_results = local_df.to_numpy().astype('float')

    if verbose == True:
        message = "_"*8+f"{common_type}"+"_"*8
        print(message)
        print(f"The column type: {common_type}")
        print(f"The resulting dataframe_shape: {np_results.shape}")
        print(f"Number of calibrating steps:{np_results.shape[0]}")
        print(f"Number of measurement by step: {np_results.shape[1]}")
        print(f"Maximum {np.max(np_results)}")
        print(f"Minimum {np.min(np_results)}")
        print("_"*len(message))
    return np_results


def process_path(cmd : Command):
    planned_path = []
    matrice_pose_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range (cmd.step_nb):
        matrice_pose_init = matrice_pose_init @ cmd.cmd_matrix
        theta_z = np.arctan2(matrice_pose_init[1, 0], matrice_pose_init[0, 0])
        # [x, y, yaw]
        planned_path.append([matrice_pose_init[0, 2], matrice_pose_init[1, 2], theta_z])
    return planned_path


def setup_axis(ax, limits):
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_xlim(limits, -limits)
    ax.set_ylim(-limits, limits)
    ax.set_aspect('equal')


def draw_path(ax, limits, path, color='b'):
    setup_axis(ax, limits)
    path = np.array(path)
    # Plot the positions of the robot as a scatter plot
    #ax.scatter(path[:, 1], path[:, 0], color=color, label='Robot position')
    # Find the u,v vectors of the orientation using the yaw angle
    u = np.cos(path[:, 2])
    v = np.sin(path[:, 2])
    # Plot the orientation of the robot as a quiver plot with only 1 point out of 10
    ax.quiver(path[::10, 1], path[::10, 0], v[::10], u[::10], angles='xy', scale_units='xy', scale=1, color=color, label='Robot pose')


def draw_path_from_command(ax, limits, cmd, color='r'):
    planned_path = process_path(cmd)
    planned_path = np.array(planned_path)
    draw_path(ax, limits, planned_path, color)


def extract_data_from_dataframe(df):
    data_dict = {}
    data_dict["cmd_vel_x"] = df['cmd_body_x_lwmean'].values
    data_dict["cmd_vel_yaw"] = df['cmd_body_yaw_lwmean'].values
    data_dict["max_lin_speed"] = df['max_linear_speed_sampled'].values
    data_dict["icp_x"] = column_type_extractor(df, "step_frame_interpolated_icp_x")
    data_dict["icp_y"] = column_type_extractor(df, "step_frame_interpolated_icp_y")
    data_dict["icp_yaw"] = column_type_extractor(df, "step_frame_interpolated_icp_yaw")
    data_dict["init_tf_x"] = df['init_tf_pose_x'].values
    data_dict["init_tf_y"] = df['init_tf_pose_y'].values
    data_dict["init_tf_yaw"] = df['init_tf_pose_yaw'].values
    return data_dict


def plot_dataframe(df, range_limit=RANGE_LIMIT):
    # Find every terrain in the dataframe
    terrains = df['terrain'].unique()
    for terrain in terrains:
        # Create a subfolder for the terrain if it does not exist with all the subfolders
        os.makedirs(f"tests_figures/{terrain}/planned_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/executed_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/combined_path", exist_ok=True)
        # Filter the dataframe to keep only the rows corresponding to the current terrain
        df_terrain = df[df['terrain'] == terrain]
        # Read the number of rows in the dataframe to get the number of commands
        cmd_nbr = df_terrain.shape[0]

        # Get the required data from the dataframe
        data_dict = extract_data_from_dataframe(df_terrain)

        # Final figures for the planned and executed paths
        fig_final, axs_final = plt.subplots(1, 2)

        # Loop over the rows of the dataframe
        for cmd_vel, cmd_angle, max_linear_speed, i in zip(data_dict["cmd_vel_x"], data_dict["cmd_vel_yaw"], data_dict["max_lin_speed"], range(cmd_nbr)):
            # Create a figure for the planned path
            fig_cmd, axs_cmd = plt.subplots(1, 1)
            # Create a figure for the executed path
            fig_icp, axs_icp = plt.subplots(1, 1)
            # Get the command from the dataframe
            cmd = Command(cmd_vel, cmd_angle, TIME_DELTA, NBR_STEPS)
            # Draw the path of the robot
            draw_path_from_command(axs_cmd, range_limit, cmd)
            axs_cmd.set_title(f"Command: {cmd_vel} m/s, {cmd_angle} rad/s")
            # Draw the path of the robot from the ICP data
            draw_path(axs_icp, range_limit, np.array([data_dict["icp_x"][i,:], data_dict["icp_y"][i,:], data_dict["icp_yaw"][i,:]]).T)
            axs_icp.set_title(f"Command: {cmd_vel} m/s, {cmd_angle} rad/s")

            fig_combined, axs_combined = plt.subplots(1, 1)
            draw_path_from_command(axs_combined, range_limit, cmd, color='r')
            draw_path(axs_combined, range_limit, np.array([data_dict["icp_x"][i,:], data_dict["icp_y"][i,:], data_dict["icp_yaw"][i,:]]).T, color='b')
            axs_combined.set_title(f"Command: {cmd_vel} m/s, {cmd_angle} rad/s")

            # Increase the spacing between the subplots
            fig_cmd.tight_layout()
            fig_icp.tight_layout()
            fig_combined.tight_layout()
            # Save the figure
            fig_cmd.savefig(f"tests_figures/{terrain}/planned_path/{i}.png")
            fig_icp.savefig(f"tests_figures/{terrain}/executed_path/{i}.png")
            fig_combined.savefig(f"tests_figures/{terrain}/combined_path/{i}.png")

            if i in INDEX_LIST_TO_PLOT:
                # Plot the planned path
                draw_path_from_command(axs_final[0], range_limit, cmd, color=COLOR_LIST_TO_PLOT[INDEX_LIST_TO_PLOT.index(i)])
                draw_path(axs_final[1], range_limit, np.array([data_dict["icp_x"][i,:], data_dict["icp_y"][i,:], data_dict["icp_yaw"][i,:]]).T, color=COLOR_LIST_TO_PLOT[INDEX_LIST_TO_PLOT.index(i)])
                fig_final.tight_layout()

        # Save the final figures
        axs_final[0].set_title("Planned paths")
        axs_final[1].set_title("Executed paths")
        # Remove the axis labels for the combined figure
        axs_final[1].set_ylabel("")
        # Save the final figure with high resolution
        fig_final.savefig(f"tests_figures/{terrain}/planned_paths.pdf", format='pdf', dpi=600)


def plot_relevant_paths(df, range_limit=RANGE_LIMIT):
    # Find every terrain in the dataframe
    terrains = df['terrain'].unique()
    for terrain in terrains:
        # Create a subfolder for the terrain if it does not exist with all the subfolders
        os.makedirs(f"tests_figures/{terrain}/planned_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/executed_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/combined_path", exist_ok=True)
        # Filter the dataframe to keep only the rows corresponding to the current terrain
        df_terrain = df[df['terrain'] == terrain]
        
        # Get the required data from the dataframe
        data_dict = extract_data_from_dataframe(df_terrain)

        # Final figures for the planned and executed paths
        fig_final, axs_final = plt.subplots(1, 2)

        for i in INDEX_LIST_TO_PLOT:
            # Get the command from the dataframe
            cmd = Command(data_dict["cmd_vel_x"][i], data_dict["cmd_vel_yaw"][i], TIME_DELTA, NBR_STEPS)

            # Plot the planned path
            draw_path_from_command(axs_final[0], range_limit, cmd, color=COLOR_LIST_TO_PLOT[INDEX_LIST_TO_PLOT.index(i)])
            draw_path(axs_final[1], range_limit, np.array([data_dict["icp_x"][i,:], data_dict["icp_y"][i,:], data_dict["icp_yaw"][i,:]]).T, color=COLOR_LIST_TO_PLOT[INDEX_LIST_TO_PLOT.index(i)])
            fig_final.tight_layout()

        # Add the title to the final figure
        axs_final[0].set_title("Planned paths")
        axs_final[1].set_title("Executed paths")
        # Save the final figure with high resolution
        fig_final.savefig(f"tests_figures/{terrain}/planned_vs_executed_paths.pdf", format='pdf', dpi=600)


def plot_heatmap_world_position(df, range_limit=RANGE_LIMIT):
    # Find every terrain in the dataframe
    terrains = df['terrain'].unique()
    for terrain in terrains:
        # Create a subfolder for the terrain if it does not exist with all the subfolders
        os.makedirs(f"tests_figures/{terrain}/planned_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/executed_path", exist_ok=True)
        os.makedirs(f"tests_figures/{terrain}/combined_path", exist_ok=True)
        # Filter the dataframe to keep only the rows corresponding to the current terrain
        df_terrain = df[df['terrain'] == terrain]

        # Read the number of rows in the dataframe to get the number of commands
        cmd_nbr = df_terrain.shape[0]

        # Get the required data from the dataframe
        data_dict = extract_data_from_dataframe(df_terrain)

        # Final figures for the planned and executed paths
        fig_final, axs_final = plt.subplots(1, 2)

        # Loop over the rows of the dataframe
        for cmd_vel, cmd_angle, max_linear_speed, i in zip(data_dict["cmd_vel_x"], data_dict["cmd_vel_yaw"], data_dict["max_lin_speed"], range(cmd_nbr)):
            # Get the command from the dataframe
            cmd = Command(cmd_vel, cmd_angle, TIME_DELTA, NBR_STEPS, [data_dict["init_tf_x"][i], data_dict["init_tf_y"][i], data_dict["init_tf_yaw"][i]])
            
    return

### Test the functions
cmd_vel = 0.0
cmd_angle = 5.0
delta_s = TIME_DELTA
step_nb = 120
cmd = Command(cmd_vel, cmd_angle, delta_s, step_nb)
fig, ax = plt.subplots()    
draw_path_from_command(ax, 30, cmd)
plt.savefig("tests_figures/planned_path_single.png")


# Load the dataframe
df_all_terrain = pd.read_pickle(PATH_DATAFRAME)

step_frame_interpolated_icp_x = column_type_extractor(df_all_terrain, "step_frame_interpolated_icp_x")
step_frame_interpolated_icp_y = column_type_extractor(df_all_terrain, "step_frame_interpolated_icp_y")
step_frame_interpolated_icp_yaw = column_type_extractor(df_all_terrain, "step_frame_interpolated_icp_yaw")

fig, ax = plt.subplots()
draw_path(ax, 30, np.array([step_frame_interpolated_icp_x[0,:], step_frame_interpolated_icp_y[0,:], step_frame_interpolated_icp_yaw[0,:]]).T)
plt.savefig("tests_figures/planned_path_interpolated.png")

# Plot the path of the robot for each command in the dataframe
#plot_dataframe(df_all_terrain)

# Plot the relevant paths for the figure
plot_relevant_paths(df_all_terrain)