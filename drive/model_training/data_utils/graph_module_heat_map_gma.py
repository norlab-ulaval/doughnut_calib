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

DATASET_PICKLE = "./drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_max_lin_speed_all_speed_all_terrain_steady_state_dataset.pkl"
GEOM_PICKLE = "./drive_datasets/results_multiple_terrain_dataframe/geom_limits_by_terrain_for_filtered_cleared_path_warthog_max_lin_speed_all_speed_all_terrain_steady_state_dataset.pkl"
NORMALIZE = True
DEBUG = True
CLINE = True

# Gaussian parameters
MU_X = 0
MU_Y = 0
SIGMA_X = 0.8
SIGMA_Y = 0.8
RHO = 0

# Vmax for the colorbar
VMAX_TRANSLATION = 1.5
VMAX_ROTATION = 1.5

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
    x,y = geom.exterior.xy
    ax.plot(x,y,color="black")

def plot_image(ax, X_train, mean_prediction, col, y, terrain, x_2_eval,
               normalize = {"normalize":False}, cline = False, filter = {},
               shape = (100,100), colormap = "PuOr", x_lim = (-6,6), y_lim = (-6,6),
               vmax = 6, cline_factor = 2):
    
    if ax == None:
        fig, ax = plt.subplots(1,1)

    if normalize["normalize"]:
        normalizer = normalize[col]
        #mean_prediction = normalizer(mean_prediction)
        #y = normalizer(y)

        if isinstance(filter,np.ndarray): 
            filtered_prediction = np.where(filter,mean_prediction.reshape(shape),0)
            
            #filtered_prediction_2d = filtered_prediction.reshape((filtered_shape,filtered_shape))

            im = ax.imshow(filtered_prediction,extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, vmin=-vmax, vmax=vmax)
            #scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
            final_shape = int(np.sqrt(mean_prediction.shape[0]))

        else:
            im = ax.imshow(mean_prediction.reshape(shape),extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, vmin=-vmax, vmax=vmax)
            #scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
            final_shape = int(np.sqrt(mean_prediction.shape[0]))

        
        if cline:
            round_vmax = np.round(vmax)
            CS = ax.contour(x_2_eval[:,0].reshape((final_shape,final_shape)),
                    x_2_eval[:,1].reshape((final_shape,final_shape)),
                    mean_prediction.reshape((final_shape,final_shape)),
                    np.linspace(-round_vmax,round_vmax, int((cline_factor*round_vmax)+1)),
                    colors="black", linewidths=0.5)
            
            ax.clabel(CS, inline=True, fontsize=10)
    else:
        if filter != {}: 
            im = ax.imshow(mean_prediction.reshape(shape)[filter],extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, vmin=-vmax, vmax=vmax)
            scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
            final_shape = int(np.sqrt(mean_prediction.shape[0]))

        else:
            im = ax.imshow(mean_prediction.reshape(shape),extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap=colormap, vmin=-vmax, vmax=vmax)
            scatter = ax.scatter(X_train[:,0],X_train[:,1],c=y,cmap=colormap,edgecolor='black',vmin=-vmax, vmax=vmax)
            final_shape = int(np.sqrt(mean_prediction.shape[0]))


        if cline:
            round_vmax = np.round(vmax)
            CS = ax.contour(x_2_eval[:,0].reshape((final_shape,final_shape)),
                       x_2_eval[:,1].reshape((final_shape,final_shape)),
                       mean_prediction.reshape((final_shape,final_shape)),
                       np.linspace(-round_vmax,round_vmax, int((cline_factor*round_vmax)+1)),
                       colors="black", linewidths=0.5)
            ax.clabel(CS, inline=True, fontsize=10)
   
    return im

def find_the_surface(df, norm, col_interest,terrain,geom_to_filter = {}, 
                     ax_mean = None, ax_std = None, debug = False, 
                     cline = False, colormap = "seismic", to_plot = "mean",
                     col_x_y = ["cmd_right_wheels","cmd_left_wheels"], x_lim = (-6,6),
                     y_lim = (-6,6), alpha = 0.7, vmax = None):
    
    # Extract the values 
    vx = np.ravel(column_type_extractor(df,col_x_y[1])) # y
    vyaw = np.ravel(column_type_extractor(df,col_x_y[0])) # x

    y = np.ravel(column_type_extractor(df,col_interest))
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

    if debug:
        #graph_scatter_valid(vx,vyaw,X,y)
        if ax_mean is not None:
            im_mean = plot_image(ax_mean, X, data_mean, col_interest, y, terrain,x_2_eval,
                        normalize = norm, cline = cline, filter = filter, shape = X_2do.shape,
                        colormap = colormap, x_lim = x_lim, y_lim = y_lim, vmax = vmax, cline_factor=4)
        if ax_std is not None:
            im_std = plot_image(ax_std, X, data_std, col_interest, y, terrain,x_2_eval,
                        normalize = norm, cline = cline, filter = filter, shape = X_2do.shape,
                        colormap = colormap, x_lim = x_lim, y_lim = y_lim, vmax = vmax, cline_factor=2)

    return im_mean, im_std

def plot_heat_map_gaussian_moving_average(data_path, geom_path, normalize = True, debug = True, cline = True):
    
    with open(geom_path, 'rb') as file:
        geom_by_terrain = pickle.load(file)["body"]

    df = pd.read_pickle(data_path)

    list_terrain = list(df.terrain.unique())
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

    norm_slip_x_ss = plt.Normalize(vmin=np.min(df.slip_body_x_ss), vmax=np.max(df.slip_body_x_ss))
    norm_slip_y_ss = plt.Normalize(vmin=np.min(df.slip_body_y_ss), vmax=np.max(df.slip_body_y_ss))
    norm_slip_yaw_ss = plt.Normalize(vmin=np.min(df.slip_body_y_ss), vmax=np.max(df.slip_body_yaw_ss))

    norm_global_dict = {"normalize": normalize,
                        "slip_body_x_ss": norm_slip_x_ss,
                        "slip_body_y_ss": norm_slip_y_ss,
                        "slip_body_yaw_ss": norm_slip_yaw_ss}
    
    for i in range(size):  
        terrain = list_terrain[i]
        
        df_terrain = df.loc[df["terrain"]==terrain]
        
        if size == 1:
            ax_to_plot_mean_1 = axs_mean[0]
            ax_to_plot_mean_2 = axs_mean[1]
            ax_to_plot_mean_3 = axs_mean[2]
            ax_to_plot_std_1 = axs_std[0]
            ax_to_plot_std_2 = axs_std[1]
            ax_to_plot_std_3 = axs_std[2]
        else:
            ax_to_plot_mean_1 = axs_mean[0,i]
            ax_to_plot_mean_2 = axs_mean[1,i]
            ax_to_plot_mean_3 = axs_mean[2,i]
            ax_to_plot_std_1 = axs_std[0,i]
            ax_to_plot_std_2 = axs_std[1,i]
            ax_to_plot_std_3 = axs_std[2,i]
                        
            geom = geom_by_terrain[terrain]
            plot_losange_limits(ax_to_plot_mean_1,geom)
            plot_losange_limits(ax_to_plot_mean_2,geom)
            plot_losange_limits(ax_to_plot_mean_3,geom)
            plot_losange_limits(ax_to_plot_std_1,geom)
            plot_losange_limits(ax_to_plot_std_2,geom)
            plot_losange_limits(ax_to_plot_std_3,geom)

        col_x_y = ["cmd_body_yaw_lwmean","cmd_body_x_lwmean" ]
        vmax_translation = max(abs(norm_global_dict["slip_body_x_ss"].vmin), norm_global_dict["slip_body_x_ss"].vmax,
                                abs(norm_global_dict["slip_body_y_ss"].vmin), norm_global_dict["slip_body_y_ss"].vmax)
        vmax_rotation = max(abs(norm_global_dict["slip_body_yaw_ss"].vmin), norm_global_dict["slip_body_yaw_ss"].vmax)

        # TODO WLH: Remove this hardcode
        # Set the vmax for the colorbar to fit the computed gma instead of the raw data
        vmax_translation = 1.5

        im_mean_1, im_std_1 = find_the_surface(df_terrain, norm_global_dict, "slip_body_x_ss",
                                                terrain, geom_to_filter = geom_by_terrain,
                                                ax_mean = ax_to_plot_mean_1, ax_std = ax_to_plot_std_1, 
                                                debug = debug, cline = cline,  colormap = "PuOr" , 
                                                col_x_y = col_x_y, vmax = vmax_translation)
        im_mean_2, im_std_2 = find_the_surface(df_terrain, norm_global_dict, "slip_body_y_ss", 
                                                terrain, geom_to_filter = geom_by_terrain,
                                                ax_mean = ax_to_plot_mean_2, ax_std = ax_to_plot_std_2, 
                                                debug = debug, cline = cline, 
                                                colormap = "PuOr", col_x_y = col_x_y, vmax = vmax_translation)
        im_mean_3, im_std_3 = find_the_surface(df_terrain,  norm_global_dict, "slip_body_yaw_ss",
                                                terrain, geom_to_filter = geom_by_terrain,
                                                ax_mean = ax_to_plot_mean_3, ax_std = ax_to_plot_std_3, 
                                                debug = debug, cline = cline, 
                                                colormap = "PiYG", col_x_y = col_x_y, vmax = vmax_rotation)

        ax_to_plot_mean_1.set_title(f"{terrain}")
        ax_to_plot_std_1.set_title(f"{terrain}")
        #ax.set_title(f"{col} on {terrain} ")
        ax_to_plot_mean_3.set_xlabel("Angular velocity [rad/s]")
        ax_to_plot_std_3.set_xlabel("Angular velocity [rad/s]")

        if i == 0:
            ax_to_plot_mean_1.set_ylabel("Linear velocity [m/s]")
            ax_to_plot_mean_2.set_ylabel("Linear velocity [m/s]")
            ax_to_plot_mean_3.set_ylabel("Linear velocity [m/s]")
            ax_to_plot_std_1.set_ylabel("Linear velocity [m/s]")
            ax_to_plot_std_2.set_ylabel("Linear velocity [m/s]")
            ax_to_plot_std_3.set_ylabel("Linear velocity [m/s]")

    if size == 1:
        # Add a colorbar
        cbar = plt.colorbar(im_mean_1, ax=axs_mean[0])
        cbar.set_label("slip body x ss [m/s]")  
        cbar = plt.colorbar(im_mean_2, ax=axs_mean[1])
        cbar.set_label("slip body y ss [m/s]")
        cbar = plt.colorbar(im_mean_3, ax=axs_mean[2])
        cbar.set_label("slip body yaw ss [rad/s]")
        cbar = plt.colorbar(im_std_1, ax=axs_std[0])
        cbar.set_label("slip body x ss [m/s]")
        cbar = plt.colorbar(im_std_2, ax=axs_std[1])
        cbar.set_label("slip body y ss [m/s]")
        cbar = plt.colorbar(im_std_3, ax=axs_std[2])
        cbar.set_label("slip body yaw ss [rad/s]")
    else:
        # Add a colorbar
        cbar = plt.colorbar(im_mean_1, ax=axs_mean[0,axs_mean.shape[1]-1])
        cbar.set_label("slip body x ss [m/s]")  
        cbar = plt.colorbar(im_mean_2, ax=axs_mean[1,axs_mean.shape[1]-1])
        cbar.set_label("slip body y ss [m/s]")
        cbar = plt.colorbar(im_mean_3, ax=axs_mean[2,axs_mean.shape[1]-1])
        cbar.set_label("slip body yaw ss [rad/s]")
        cbar = plt.colorbar(im_std_1, ax=axs_std[0,axs_std.shape[1]-1])
        cbar.set_label("slip body x ss [m/s]")
        cbar = plt.colorbar(im_std_2, ax=axs_std[1,axs_std.shape[1]-1])
        cbar.set_label("slip body y ss [m/s]")
        cbar = plt.colorbar(im_std_3, ax=axs_std[2,axs_std.shape[1]-1])
        cbar.set_label("slip body yaw ss [rad/s]")  
        
    for ax in np.ravel(axs_mean):
        ax.set_facecolor("black")
        ax.set_aspect('equal', 'box')

    for ax in np.ravel(axs_std):
        ax.set_facecolor("black")
        ax.set_aspect('equal', 'box')

    # Optional label for the colorbar
    fig_mean.savefig(path.parent/("mean" + path.parts[-1]+".pdf"),format="pdf")
    fig_std.savefig(path.parent/("std" + path.parts[-1]+".pdf"),format="pdf")
    plt.show()


if __name__=="__main__":
    # Arg parser for the pickle file
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--dataset",type=str,help="Path to the pickle file", default=DATASET_PICKLE)
    parser.add_argument("--geom",type=str,help="Path to the pickle file containing the geom", default=GEOM_PICKLE)
    parser.add_argument("--normalize",type=bool,help="Normalize the data", default=NORMALIZE)
    parser.add_argument("--debug",type=bool,help="Debug mode", default=DEBUG)
    parser.add_argument("--cline",type=bool,help="Plot the cline", default=CLINE)

    # Parse the arguments
    args = parser.parse_args()

    path = pathlib.Path(parser.parse_args().dataset)
    path_to_geom = pathlib.Path(parser.parse_args().geom)
    normalize = parser.parse_args().normalize
    debug = parser.parse_args().debug
    cline = parser.parse_args().cline

    plot_heat_map_gaussian_moving_average(path, path_to_geom, normalize, debug, cline)