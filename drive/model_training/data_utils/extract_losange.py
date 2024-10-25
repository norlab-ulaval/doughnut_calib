import pandas as pd 
import shapely 
from drive.model_training.data_utils.extractors import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pathlib
from shapely import concave_hull,MultiPoint
import pickle

def plot_command_space(df,ax):
    alpha_parama = 0.6
    ax_to_plot = ax
    ax_to_plot.scatter(df["cmd_body_yaw_lwmean"],df["cmd_body_x_lwmean"],color = "orange",label='Command',alpha=alpha_parama)
    ax_to_plot.set_xlabel("Angular velocity (omega) [rad/s]")
    ax_to_plot.set_ylabel("Forward velocity (V_x) [m/s]")
    
    ax,area = extract_minimum_sphere(ax,df["cmd_body_yaw_lwmean"].to_numpy(),df["cmd_body_x_lwmean"].to_numpy(),"orange")
    
    return ax, area

def plot_diamond(df,ax,color,no_scatter=True):

    alpha_parama = 1
    if no_scatter ==False:
        ax.scatter(df["icp_vel_yaw_smoothed"],df["icp_vel_x_smoothed"],color = color,label='Mean of body steady-state speed',alpha=alpha_parama) 

    ax,area = extract_minimum_sphere(ax,df["icp_vel_yaw_smoothed"].to_numpy(),df["icp_vel_x_smoothed"].to_numpy(),color)
    
    return ax,area

def extract_minimum_sphere(ax,x,y,color):

    points = np.concat((x.reshape((x.shape[0],1)),y.reshape((x.shape[0],1))),axis=1)

    points_shapely = MultiPoint(points)
    polygon = concave_hull(points_shapely,ratio=0.3)
    print(polygon)
    area = polygon.area
    x,y = polygon.exterior.xy
    ax.fill(x, y, color=color, alpha=0.1,label="convex hull")  # Fill color with transparency
    #hull = ConvexHull(points)
    #area = hull.area
    #ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.1,label="convex hull")  # Fill color with transparency

    return ax, area 

def extract_x_y_area(x,y):

    points = np.concat((x.reshape((x.shape[0],1)),y.reshape((x.shape[0],1))),axis=1)
    points_shapely = MultiPoint(points)
    polygon = concave_hull(points_shapely,ratio=0.3)
    area = polygon.area
    x,y = polygon.exterior.xy
    x_y=np.array([x,y]).T
    return x_y,area

def extract_x_y_area_all_terrains(df, prefix=""):
    """Extract the point contouring the cmd and the terrain and then export a json file containting this point 

    Args:
        df (_type_): dataframe with the folowing columns [cmd_body_yaw_lwmean, cmd_body_x_lwmean, icp_vel_yaw_smoothed,icp_vel_x_smoothed,
                    cmd_right_wheels, cmd_left_wheels,odom_speed_right_wheels,odom_speed_left_wheels ]
        path_to_save (_type_): pathlib_object pointing to the results. 

    Returns:
        __dict__: dict in the folowing format {"terrain1":{"cmd_body_contour_points": ,"speed_body_contour_points": ,
        "wheel_cmd": , "wheel_gt": }, terrain2":{...}   }
    """
    list_terrain = list(df.terrain.unique())
    list_area_data = []
    dico_results = {}
    for terrain in list_terrain:

        df_terrain = df.loc[df["terrain"] == terrain]

        
        countours_points_cmd, area_cmd = extract_x_y_area(df_terrain["cmd_body_yaw_lwmean"].to_numpy(),df_terrain["cmd_body_x_lwmean"].to_numpy())
        countours_points_gt, area_gt = extract_x_y_area(df_terrain["icp_vel_yaw_smoothed"].to_numpy(),df_terrain["icp_vel_x_smoothed"].to_numpy())
        countours_points_wheel_cmd, wheel_cmd = extract_x_y_area(df_terrain["cmd_right_wheels"].to_numpy(),df_terrain["cmd_left_wheels"].to_numpy())
        countours_points_wheel_gt, wheel_gt = extract_x_y_area(df_terrain["odom_speed_right_wheels"].to_numpy(),df_terrain["odom_speed_left_wheels"].to_numpy())
        
        dico_results[terrain] = {"cmd_body_contour_points": countours_points_cmd,
                                "speed_body_contour_points": countours_points_gt,
                                "wheel_cmd": countours_points_wheel_cmd,
                                "wheel_gt": countours_points_wheel_gt
                                }
        
        list_area_data.append([terrain,area_cmd,area_gt,wheel_cmd,wheel_gt])
    # 
    path_to_dir = pathlib.Path("drive_datasets/results_multiple_terrain_dataframe/terrain_zone")
    if path_to_dir.is_dir()==False:
        path_to_dir.mkdir()

    # Specify the filename
    filename = path_to_dir/(prefix+'terrain_contour_dict.pkl')

    # Write the dictionary to a JSON file
    with open(str(filename), 'wb') as json_file:
        pickle.dump(dico_results, json_file)

    print(f"Area are saved at the following path: {filename}")
    dico_results["path_to_results"] = filename 


    #Save results
    
    df = pd.DataFrame(data=list_area_data,columns=["terrain","body_cmd_area","body_res_area", "wheel_cmd_area","wheel_res_area" ])
    df.to_pickle(path_to_dir/(prefix+'terrain_area_df.pkl'))


    return dico_results
        
    
        
        
def get_area_coverage(path_to_saved_json_file,show=False):

    color_dict = {"asphalt":"grey", "ice":"blue","gravel":"brown","grass":"green","tile":"pink","boreal":"lightgray","sand": "orange"}


    with open(path_to_saved_json_file, 'rb') as json_file:
        data = pickle.load(json_file)

    if show:
        fig, ax = plt.subplots(2,1)

        for terrain, dict in data.items():

            
            ax[0].plot(dict["cmd_body_contour_points"][:,0],dict["cmd_body_contour_points"][:,1],label="cmd "+terrain,color=color_dict[terrain],ls="--")
            ax[0].plot(dict["speed_body_contour_points"][:,0],dict["speed_body_contour_points"][:,1],label="gt "+terrain,color=color_dict[terrain])

            ax[1].plot(dict["wheel_cmd"][:,0],dict["wheel_cmd"][:,1],label="cmd "+terrain,color=color_dict[terrain],ls="--")
            ax[1].plot(dict["wheel_gt"][:,0],dict["wheel_gt"][:,1],label="gt "+terrain,color=color_dict[terrain])


        ax[0].set_xlabel("Angular speed body [rad/s]")
        ax[0].set_ylabel("Linear speed body [m/s]")

        ax[1].set_xlabel("Left wheel [rad/s]")
        ax[1].set_ylabel("Right wheel [rad/s]")

        
        ax[1].legend()
        ax[1].set_aspect("equal",adjustable="datalim")

        plt.show()
    return data

if __name__ == "__main__":

    
    
    
    steady_state_path = "drive_datasets/results_multiple_terrain_dataframe/all_terrain_steady_state_dataset.pkl"
    df_diamond = pd.read_pickle(steady_state_path)


    df_warthog = df_diamond.loc[df_diamond["robot"]=="warthog"]
    df_sampling_speed = df_warthog.loc[df_warthog["max_linear_speed_sampled"]==5.0]
    
    dico_results = extract_x_y_area_all_terrains(df_sampling_speed, prefix="")   
    
    get_area_coverage(dico_results["path_to_results"],show=True)

    color_dict = {"asphalt":"grey", "ice":"blue","gravel":"brown","grass":"green","tile":"pink","boreal":"lightgray","sand": "linen"}

    # Create the folder of results 
    path_to_save_data = pathlib.Path("drive_datasets/results_multiple_terrain_dataframe/area_test")
    if path_to_save_data.is_dir == False:
        path_to_save_data.mkdir()
    
    print_column_unique_column(df_diamond)
    list_terrain = list(df_sampling_speed.terrain.unique())

    # INstall the ax
    
    

    list_results_area = []
    lists_command_area = []
    
    
    for terrain in list_terrain:
        fig, ax = plt.subplots(1,1)
        ax.set_title(terrain)
        df_terrain = df_sampling_speed.loc[df_sampling_speed.terrain==terrain]
        ax, area_command = plot_command_space(df_terrain,ax)
        ax, area_terrain = plot_diamond(df_terrain,ax,color_dict[terrain],no_scatter=False)
        
        list_results_area.append(area_terrain)
        lists_command_area.append(area_command)
        ax.legend()
        plt.show()

    dict_= {"terrain":list_terrain,"commanded_area":lists_command_area,"resulting_area":list_results_area}

    df = pd.DataFrame.from_dict(dict_)
    df.to_pickle(path_to_save_data/"areas_saved.pkl")



    list_results_area = []
    lists_command_area = []
    fig, ax = plt.subplots(1,1)
    ax.set_title("All zone")
    for terrain in list_terrain:
        
        df_terrain = df_sampling_speed.loc[df_sampling_speed.terrain==terrain]
        ax, area_command = plot_command_space(df_terrain,ax)
        ax, area_terrain = plot_diamond(df_terrain,ax,color_dict[terrain],no_scatter=True)
        
        list_results_area.append(area_terrain)
        list_results_area.append(area_command)
        
    ax.legend()
    plt.show()
    dict_= {"terrain":list_terrain,"commanded_area":lists_command_area,"resulting_area":list_results_area}

    
    print(df)

    


