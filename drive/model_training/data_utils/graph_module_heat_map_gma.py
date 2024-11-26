import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from drive.model_training.data_utils.extractors import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel
import matplotlib.cm as cm 
import pickle 
import shapely
import pathlib
import argparse
import matplotlib.colors as mcolors

ROBOT = "husky"
if ROBOT == "husky":
    DATASET_PICKLE = "drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_husky_following_robot_param_all_terrain_steady_state_dataset.pkl"
    GEOM_PICKLE = "drive_datasets/results_multiple_terrain_dataframe/husky_geom_limits_by_terrain_for_filtered_cleared_path_husky_following_robot_param_all_terrain_steady_state_dataset.pkl"
    AXIS_LIM = (-2,2)
elif ROBOT == "warthog":
    DATASET_PICKLE = "drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_following_robot_param_all_terrain_steady_state_dataset.pkl"
    GEOM_PICKLE = "drive_datasets/results_multiple_terrain_dataframe/warthog_geom_limits_by_terrain_for_filtered_cleared_path_warthog_following_robot_param_all_terrain_steady_state_dataset.pkl"
    AXIS_LIM = (-6,6)
TOGGLE_CLINE = True
TOGGLE_PROPORTIONNAL = False
LIST_OF_TERRAINS_TO_PLOT = ["grass","gravel","mud","sand","ice","asphalt", "tile"]

# Gaussian parameters
MU_X = 0
MU_Y = 0
SIGMA_X = 0.8
SIGMA_Y = 0.8
RHO = 0

# List of cline factor
LIST_CLINE_FACTOR = [2, 2, 1]

def gaussian_2d(x, y, mu_x=MU_X, mu_y=MU_Y, sigma_x=SIGMA_X, sigma_y=SIGMA_Y, rho=RHO):
    norm_const = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    exp_part = -1 / (2 * (1 - rho**2)) * (
        ((x - mu_x)**2 / sigma_x**2) + 
        ((y - mu_y)**2 / sigma_y**2) - 
        (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    )
    return norm_const * np.exp(exp_part)


def filtered_mesgrid_cmd_based_on_geom(geom_by_terrain,terrain,x_mesh,y_mesh):

    geom = geom_by_terrain[terrain]

    list_fitler = []

    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)

    for x,y in zip(x_ravel,y_ravel):
        pt = shapely.geometry.Point(x,y)
        if shapely.within(pt,geom):
            list_fitler.append(True)
        else:
            list_fitler.append(False)

    filter = np.array(list_fitler).reshape(x_mesh.shape)

    return filter


def plot_losange_limits(ax,geom):
    # If ax is an array of ax then plot the losange on each ax
    if isinstance(ax,np.ndarray):
        for ax_ in ax:
            plot_losange_limits(ax_,geom)
    else:
        x,y = geom.exterior.xy
        ax.plot(x,y, color="black")


def plot_image(ax, X_train, mean_prediction, y, x_2_eval, cline_factor = None, filter = {},
               shape = (100,100), colormap = "PuOr", x_lim = AXIS_LIM, y_lim = AXIS_LIM,
               vmax = 6, proportionnal = False):
    if ax == None:
        fig, ax = plt.subplots(1,1)

    # Change the axis to be log scale
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    print(f"vmax: {vmax}")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    if proportionnal:
        norm = mcolors.LogNorm(vmin=0.1, vmax=vmax)

    if isinstance(filter,np.ndarray): 
        filtered_prediction = np.where(filter,mean_prediction.reshape(shape),0)
        
        im = ax.imshow(filtered_prediction,extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, norm=norm)
        #scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
        final_shape = int(np.sqrt(mean_prediction.shape[0]))

    else:
        im = ax.imshow(mean_prediction.reshape(shape),extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, norm=norm)
        #scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
        final_shape = int(np.sqrt(mean_prediction.shape[0]))

    
    if cline_factor is not None:
        round_vmax = np.round(vmax)
        CS = ax.contour(x_2_eval[:,0].reshape((final_shape,final_shape)),
                x_2_eval[:,1].reshape((final_shape,final_shape)),
                mean_prediction.reshape((final_shape,final_shape)),
                #np.linspace(-round_vmax,round_vmax, int((cline_factor*round_vmax)+1)),
                [-1, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1],
                colors="black", linewidths=0.5)
        
        ax.clabel(CS, inline=True, fontsize=10)

    return im

def process_gma_meshgrid(X, y, x_2_eval):
    slip_list_mean = []
    slip_list_std = []
    # Compute the slip for the meshgrid
    for i in x_2_eval:
        # Recenter all the values around the point i
        X_centered = X - i
        # Compute the amplitude of the point X_centered with the gaussian function
        amplitude = gaussian_2d(X_centered[:,0],X_centered[:,1])
        slip = np.sum(amplitude*y)/np.sum(amplitude)
        slip_list_mean.append(slip)
        relative_y = y - slip
        slip_list_std.append(np.sqrt(np.sum(amplitude*(relative_y**2))/np.sum(amplitude)))

    data_mean = np.array(slip_list_mean)
    data_std = np.array(slip_list_std)

    return data_mean, data_std


def process_data(df, list_col_interest,terrain,geom_to_filter = {}, 
                list_colormap = None, col_x_y = ["cmd_right_wheels","cmd_left_wheels"],
                x_lim = AXIS_LIM, y_lim = AXIS_LIM, proportionnal = False):
    
    # Assert that the number of element in list_ax_mean, list_ax_std and list_col_interest are the same
    assert len(list_col_interest) == len(list_colormap)

    # Extract the values 
    vx = np.ravel(column_type_extractor(df,col_x_y[1])) # y
    vyaw = np.ravel(column_type_extractor(df,col_x_y[0])) # x
    X = np.array((vyaw,vx)).T

    # Predict the grid 
    # Define the ranges for x and y
    x_lim_to_plot = np.linspace(x_lim[0], x_lim[1], 100)  # 100 points from -5 to 5
    y_lim_to_plot = np.linspace(y_lim[0], y_lim[1], 100)  # 100 points from -5 to 5

    # Create the meshgrid
    X_2do, Y_2do = np.meshgrid(x_lim_to_plot, y_lim_to_plot)

    if geom_to_filter != {}:
        filter = filtered_mesgrid_cmd_based_on_geom(geom_to_filter,terrain,X_2do,Y_2do)
    else:
        filter = {} 

    x_2_eval = np.array((np.ravel(X_2do),np.ravel(Y_2do))).T

    # Create a list for the data mean and std
    list_data_mean = []
    list_data_std = []
    list_y = []

    for i in range(len(list_col_interest)):
        y = np.ravel(column_type_extractor(df, list_col_interest[i]))
        data_mean, data_std = process_gma_meshgrid(X, y, x_2_eval)
        if proportionnal:
            # If list_col_interest contains yaw in the name then we need to divide by the angular velocity
            if "yaw" in list_col_interest[i]:
                data_mean = abs(data_mean/np.ravel(X_2do))
                data_std = abs(data_std/np.ravel(X_2do))
            else:
                data_mean = abs(data_mean/np.ravel(Y_2do))
                data_std = abs(data_std/np.ravel(Y_2do))
        list_data_mean.append(data_mean)
        list_data_std.append(data_std)
        list_y.append(y)
    
    # Create a dictionary to store the results
    dict_results = {}
    dict_results["list_data_mean"] = list_data_mean
    dict_results["list_data_std"] = list_data_std
    dict_results["list_y"] = list_y
    dict_results["X"] = X
    dict_results["x_2_eval"] = x_2_eval
    dict_results["X_2do"] = X_2do
    dict_results["x_lim"] = x_lim
    dict_results["y_lim"] = y_lim
    dict_results["filter"] = filter
    return dict_results


def plot_heat_map_gaussian_moving_average(data_path, geom_path, cline = True, proportionnal = False):
    
    with open(geom_path, 'rb') as file:
        geom_by_terrain = pickle.load(file)["body"]

    df = pd.read_pickle(data_path)

    list_terrain = list(df.terrain.unique())
    # Remove any terrain that is not in the list of terrain to plot
    list_terrain = [terrain for terrain in list_terrain if terrain in LIST_OF_TERRAINS_TO_PLOT]
    size = len(list_terrain)
    fig_mean, axs_mean = plt.subplots(3,size)
    fig_std, axs_std = plt.subplots(3,size)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    fig_mean.set_figwidth(3.5*size)
    fig_mean.set_figheight(3*3)
    fig_std.set_figwidth(3.5*size)
    fig_std.set_figheight(3*3)
    fig_mean.canvas.manager.set_window_title('Mean Heat Map')
    fig_std.canvas.manager.set_window_title('Standard Deviation Heat Map')
    
    if proportionnal:
        list_col_interest = ["slip_body_x_ss","slip_body_yaw_ss"]
        list_colormap = ["PuOr", "PiYG"]
    else:
        list_col_interest = ["slip_body_x_ss","slip_body_y_ss","slip_body_yaw_ss"]
        list_colormap = ["PuOr", "PuOr", "PiYG"]
    
    # Create a list by terrain for the data mean, std and y
    terrain_dict = {}

    # Loop over the terrain
    for i in range(size):
        terrain = list_terrain[i]
        
        df_terrain = df.loc[df["terrain"]==terrain]
        col_x_y = ["cmd_body_yaw_lwmean","cmd_body_x_lwmean" ]
        
        if cline:
            list_cline_factor = LIST_CLINE_FACTOR
        else:
            list_cline_factor = [None]*len(list_col_interest)
        dict_results = process_data(df_terrain, list_col_interest, terrain, geom_to_filter = geom_by_terrain, 
                                    list_colormap = list_colormap, col_x_y = col_x_y, proportionnal = proportionnal)
        
        terrain_dict[terrain] = dict_results

    # Create a dictionary to store the vmax for each colormap
    dict_vmax_mean = {}
    dict_vmax_std = {}
    for j in range(size):
        terrain = list_terrain[j]
        for i in range(len(list_colormap)):
            mean_list = terrain_dict[terrain]["list_data_mean"][i]
            std_list = terrain_dict[terrain]["list_data_std"][i]
            vmax_mean = np.max(np.abs(mean_list[~np.isnan(mean_list)]))
            vmax_std = np.max(np.abs(std_list[~np.isnan(std_list)]))
            if list_colormap[i] in dict_vmax_mean:
                if dict_vmax_mean[list_colormap[i]] < vmax_mean:
                    dict_vmax_mean[list_colormap[i]] = vmax_mean
            else:
                dict_vmax_mean[list_colormap[i]] = vmax_mean
            if list_colormap[i] in dict_vmax_std:
                if dict_vmax_std[list_colormap[i]] < vmax_std:
                    dict_vmax_std[list_colormap[i]] = vmax_std
            else:
                dict_vmax_std[list_colormap[i]] = vmax_std

    for i in range(size):
        terrain = list_terrain[i]
        if size == 1:
            axs_mean_plot = axs_mean[:]
            axs_std_plot = axs_std[:]
        else:
            axs_mean_plot = axs_mean[:,i]
            axs_std_plot = axs_std[:,i]
                        
            geom = geom_by_terrain[terrain]
            plot_losange_limits(axs_mean_plot,geom)
            plot_losange_limits(axs_std_plot,geom)

        X = terrain_dict[terrain]["X"]
        x_2_eval = terrain_dict[terrain]["x_2_eval"]
        X_2do = terrain_dict[terrain]["X_2do"]
        x_lim = terrain_dict[terrain]["x_lim"]
        y_lim = terrain_dict[terrain]["y_lim"]
        filter = terrain_dict[terrain]["filter"]
        list_im_mean = []
        list_im_std = []
        for j in range(len(list_col_interest)):
            data_mean = terrain_dict[terrain]["list_data_mean"][j]
            data_std = terrain_dict[terrain]["list_data_std"][j]
            y = terrain_dict[terrain]["list_y"][j]
            list_im_mean.append(plot_image(axs_mean_plot[j], X, data_mean, y, x_2_eval, cline_factor = list_cline_factor[j], filter = filter,
                shape = X_2do.shape, colormap = list_colormap[j], x_lim = x_lim, y_lim = y_lim, vmax = dict_vmax_mean[list_colormap[j]], proportionnal = proportionnal))
            list_im_std.append(plot_image(axs_std_plot[j], X, data_std, y, x_2_eval, cline_factor = list_cline_factor[j], filter = filter,
                shape = X_2do.shape, colormap = list_colormap[j], x_lim = x_lim, y_lim = y_lim, vmax = dict_vmax_std[list_colormap[j]], proportionnal = proportionnal))

        axs_mean_plot[0].set_title(f"{terrain}")
        axs_std_plot[0].set_title(f"{terrain}")
        #ax.set_title(f"{col} on {terrain} ")
        axs_mean_plot[-1].set_xlabel("Angular velocity [rad/s]")
        axs_mean_plot[-1].set_xlabel("Angular velocity [rad/s]")

        if i == 0:
            for ax in axs_mean_plot:
                ax.set_ylabel("Linear velocity [m/s]")
            for ax in axs_std_plot:
                ax.set_ylabel("Linear velocity [m/s]")

    if size == 1:
        # Add a colorbar
        cbar = plt.colorbar(list_im_mean[0], ax=axs_mean_plot[0])
        cbar.set_label("Slip Body X ss [m/s]")  
        cbar = plt.colorbar(list_im_mean[1], ax=axs_mean_plot[1])
        cbar.set_label("Slip Body Y ss [m/s]")
        cbar = plt.colorbar(list_im_mean[2], ax=axs_mean_plot[2])
        cbar.set_label("Slip Body yaw ss [rad/s]")
        cbar = plt.colorbar(list_im_std[0], ax=axs_std_plot[0])
        cbar.set_label("Slip Body X ss [m/s]")
        cbar = plt.colorbar(list_im_std[1], ax=axs_std_plot[1])
        cbar.set_label("Slip Body Y ss [m/s]")
        cbar = plt.colorbar(list_im_std[2], ax=axs_std_plot[2])
        cbar.set_label("Slip Body yaw ss [rad/s]")
    else:
        # Add a colorbar
        cbar = plt.colorbar(list_im_mean[0], ax=axs_mean[0,axs_mean.shape[1]-1])
        cbar.set_label("Slip Body X ss [m/s]")  
        cbar = plt.colorbar(list_im_mean[1], ax=axs_mean[1,axs_mean.shape[1]-1])
        cbar.set_label("Slip Body Y ss [m/s]")
        cbar = plt.colorbar(list_im_mean[2], ax=axs_mean[2,axs_mean.shape[1]-1])
        cbar.set_label("Slip Body yaw ss [rad/s]")
        cbar = plt.colorbar(list_im_std[0], ax=axs_std[0,axs_std.shape[1]-1])
        cbar.set_label("Slip Body X ss [m/s]")
        cbar = plt.colorbar(list_im_std[1], ax=axs_std[1,axs_std.shape[1]-1])
        cbar.set_label("Slip Body Y ss [m/s]")
        cbar = plt.colorbar(list_im_std[2], ax=axs_std[2,axs_std.shape[1]-1])
        cbar.set_label("Slip Body yaw ss [rad/s]")
        
    for ax in np.ravel(axs_mean):
        ax.set_facecolor("black")
        ax.set_aspect('equal', 'box')

    for ax in np.ravel(axs_std):
        ax.set_facecolor("black")
        ax.set_aspect('equal', 'box')

    # Optional label for the colorbar
    mean_filename = f"mean_heat_map_gma_{ROBOT}.pdf"
    std_filename = f"std_heat_map_gma_{ROBOT}.pdf"
    fig_mean.savefig(path.parent/mean_filename,format="pdf")
    fig_std.savefig(path.parent/std_filename,format="pdf")
    plt.show()


if __name__=="__main__":
    # Arg parser for the pickle file
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--dataset",type=str,help="Path to the pickle file", default=DATASET_PICKLE)
    parser.add_argument("--geom",type=str,help="Path to the pickle file containing the geom", default=GEOM_PICKLE)
    parser.add_argument("--cline",type=bool,help="Plot the cline", default=TOGGLE_CLINE)
    parser.add_argument("--proportionnal", type=bool, help="Plot the proportionnal slip instead of the absolute", default=TOGGLE_PROPORTIONNAL)

    # Parse the arguments
    args = parser.parse_args()

    path = pathlib.Path(parser.parse_args().dataset)
    path_to_geom = pathlib.Path(parser.parse_args().geom)
    cline = parser.parse_args().cline
    proportionnal = parser.parse_args().proportionnal

    plot_heat_map_gaussian_moving_average(path, path_to_geom, cline, proportionnal)