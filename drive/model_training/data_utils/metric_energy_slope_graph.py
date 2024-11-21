import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 






from matplotlib.patches import Ellipse



def plot_metric_scatter(df_res,alpha_param=0.4,suffix="",show_ellipse=False):

    df_all = df_res.copy()
    #df.reset_index(inplace=True, names="terrain")

    # col = ["slope","slope_std","x_95","y_std_95"]

    fig, axs = plt.subplots(3,1)
    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.subplots_adjust(hspace=0.4,wspace=0.4)

    color_dict = {"asphalt":"pink", "ice":"blue","gravel":"orange","grass":"green","sand":"darkgoldenrod"}
    columns = ["rotationnal_energy_metric","translationnal_energy_metric","total_energy_metric" ]
    
    
    # Scatter plot the data points 
    list_marker = [
    'o',  # Circle
    's',  # Square
    '^',  # Triangle Up
    'v',  # Triangle Down
    '>',  # Triangle Right
    '<',  # Triangle Left
    'p',  # Pentagon
    '*',  # Star
    'H',  # Hexagon (regular)
    'h',  # Hexagon (alternate)
    '+',  # Plus
    'x',  # Cross
    'D',  # Diamond
    'd',  # Thin Diamond
    '|',  # Vertical Line
    '_',  # Horizontal Line
    ]
    i_marker = 0

    first_time_terrain = True
    for lim_yaw in df_all["lim_vel_yaw"].unique(): 
        
        df_lim_vel_yaw = df_all.loc[df_all.lim_vel_yaw == lim_yaw]
        
        for lim_vel_x in df_lim_vel_yaw["lim_vel_x"].unique(): 
            df = df_lim_vel_yaw.loc[df_lim_vel_yaw.lim_vel_x == lim_vel_x]
            
            first_time_symbol = True
            
            for column,ax in zip(columns,np.ravel(axs)):
                x = df["metric_"+column].to_numpy()
                y = df["cmd_95_"+column].to_numpy() * df["mean_slope_"+column].to_numpy() 
                color_list = [color_dict[terrain] for terrain in df.terrain]
                
                
                if first_time_terrain:
                    for i in range(len(x)):
                        ax.scatter(x[i], y[i], color=color_list[i], marker=list_marker[i_marker],label=df.terrain.loc[i])
                    first_time_terrain = False
                else:
                    ax.scatter(x, y, color=color_list, marker=list_marker[i_marker])
                
                if first_time_symbol:
                    ax.scatter(0.7, 0, color="black", marker=list_marker[i_marker], label=f"lim vel_x {lim_vel_x} \n lim vel_yaw {lim_yaw}")
                    first_time_symbol = False
                    
                # Plot uncertainty ellipses around each data point
                if show_ellipse:
                    for i in range(len(x)):

                        
                        # Covariance matrix for the uncertainty (example: variance in x and y)
                        y_std_95 = df.cmd_95_total_energy_metric.to_numpy() * df.std_slope_total_energy_metric.to_numpy() 

                        cov = np.array([[df.std_metric_total_energy_metric.loc[i]**2, 0], [0, y_std_95[i]**2]])  # Example covariance matrix, adjust as needed

                        # Create an ellipse
                        eigvals, eigvecs = np.linalg.eigh(cov)  # Eigenvalues (size of axes) and eigenvectors (orientation)
                        angle = np.arctan2(*eigvecs[:, 0][::-1])  # Angle of rotation for the ellipse
                        width, height = 1 * np.sqrt(eigvals)  # Width and height (2 * std dev, i.e., 95% confidence ellipse)
                        
                        # Create an Ellipse object
                        
                        
                        ellipse = Ellipse((x[i], y[i]), width, height, angle=np.degrees(angle), 
                                        edgecolor=color_list[i], facecolor=color_list[i], linestyle='-',
                                        label= df.terrain[i]+" 1 std",alpha = alpha_param)
                        
                        # Add the ellipse to the plot
                        ax.add_patch(ellipse)

            


                # Add labels and legend
                ax.set_xlabel("Difficulty metric (1-slope)")
                ax.set_ylabel(f"{column} @(x=95 \%) ")
                ax.set_title('New mapping of DRIVE (mass include)'+suffix)
                
                ax.grid(True)
                ylimit = ax.get_ylim()
                ax.set_ylim(0,ylimit[1])
                ax.set_xlim(0,1)

            i_marker += 1

    axs[0].legend(ncol=5)
    #plt.tight_layout()

def plot_metric_depending_on_sampling_space(df_res,suffix=""):

    df = df_res.copy()
    #df.reset_index(inplace=True, names="terrain")

    # col = ["slope","slope_std","x_95","y_std_95"]

    fig, axs = plt.subplots(3,2)
    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.subplots_adjust(hspace=0.4,wspace=0.4)

    color_dict = {"asphalt":"pink", "ice":"blue","gravel":"orange","grass":"green","sand":"darkgoldenrod"}
    columns = ["rotationnal_energy_metric","translationnal_energy_metric","total_energy_metric" ]
    
    
    # Scatter plot the data points 
    list_marker = [
    'o',  # Circle
    's',  # Square
    '^',  # Triangle Up
    'v',  # Triangle Down
    '>',  # Triangle Right
    '<',  # Triangle Left
    'p',  # Pentagon
    '*',  # Star
    'H',  # Hexagon (regular)
    'h',  # Hexagon (alternate)
    '+',  # Plus
    'x',  # Cross
    'D',  # Diamond
    'd',  # Thin Diamond
    '|',  # Vertical Line
    '_',  # Horizontal Line
    ]
    i_marker = 0

    first_time_terrain = True
    list_x_axis = ["lim_vel_x","lim_vel_yaw"]
    col_index = [0,1]
    for x_axis, col_i in zip(list_x_axis,col_index):
        
        axs_i = axs[:,col_i]
        for column,ax in zip(columns,np.ravel(axs_i)):
            y = df["metric_"+column].to_numpy()
            x = df[x_axis].to_numpy()
            color_list = [color_dict[terrain] for terrain in df.terrain]
            
            
            if first_time_terrain:
                for i in range(len(x)):
                    ax.scatter(x[i], y[i], color=color_list[i], marker=list_marker[i_marker],label=df.terrain.loc[i])
                first_time_terrain = False
            else:
                ax.scatter(x, y, color=color_list, marker=list_marker[i_marker])
            
            
            


            # Add labels and legend
            ax.set_ylabel(f"Metric based \n {column}")
            
            ax.set_title('New mapping of DRIVE (mass include)'+suffix)
            
            ax.grid(True)
            ylimit = ax.get_ylim()
            ax.set_ylim(0,ylimit[1])
            ax.set_xlim(0,5.0)
            axs[0,0].legend(ncol=5)
    axs[-1,0].set_xlabel(f"lim_vel_x")
    axs[-1,1].set_xlabel(f"lim_vel_yaw")
        
    
    

def plot_metric_3d(df_res,suffix=""):

    df_all = df_res.copy()

    
    #df.reset_index(inplace=True, names="terrain")

    # col = ["slope","slope_std","x_95","y_std_95"]
    list_terrain = list(df_all.terrain.unique())

    fig, axs = plt.subplots(3,len(list_terrain))
    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.subplots_adjust(hspace=0.4,wspace=0.4)

    color_dict = {"asphalt":"pink", "ice":"blue","gravel":"orange","grass":"green","sand":"darkgoldenrod"}
    columns = ["rotationnal_energy_metric","translationnal_energy_metric","total_energy_metric" ]
    
    list_vmin_vmax = [[df_all.metric_rotationnal_energy_metric.min(),df_all.metric_rotationnal_energy_metric.max()],
                      [df_all.metric_translationnal_energy_metric.min(),df_all.metric_translationnal_energy_metric.max()],
                      [df_all.metric_total_energy_metric.min(),df_all.metric_total_energy_metric.max()]]
    for j,terrain in enumerate(list_terrain):
        axs_j = axs[:,j]
        df = df_all.loc[df_all.terrain==terrain]
        axs_j[0].set_title(terrain)
        shape = df.lim_vel_yaw.value_counts().loc[df.lim_vel_yaw.unique()[0]]

        for col, ax,vminmax in zip(columns,np.ravel(axs_j),list_vmin_vmax):
            
            metric_value = df["metric_"+col].to_numpy().reshape(shape,shape)
            #metric_value = df["lim_vel_yaw"].to_numpy().reshape(shape,shape)
            
            pictur = ax.imshow(metric_value, cmap='viridis', interpolation='nearest',extent = [0.5, 5, 5, 0.5],
                               )
            ax.invert_yaxis()
            
            ax.set_xlabel("lim_x")
            ax.set_ylabel("lim_yaw")
            # Add color bar to indicate the values
            fig.colorbar(pictur,label=f'metric {col}',ax=ax)

            # Add titles and labels (optional)
            #plt.title('Heatmap Example')
            #plt.xlabel('X Axis')
            #plt.ylabel('Y Axis')

            # Show the plot
    
    plt.show()
        
def moving_average(x,y,percentile=95.0, r = 0.01):


    
    x_windows = np.linspace(0,1,101)
    y_windows = np.zeros_like(x_windows)
    y_windows_std = np.zeros_like(x_windows)
    for i,x_window in enumerate(x_windows):

        mask =  np.abs(x-x_window) <= r

        x_masked = x[mask]
        y_masked = y[mask]

        if y_masked.size == 0:
            continue
        else:
            y_windows[i] = np.percentile(y_masked,percentile)
            y_windows_std[i] = np.std(y_masked)
    return x_windows,y_windows,y_windows_std







def plot_metric_scatter_scatter(df_res,alpha_param=0.4,suffix="",y_column="y_coordinates",percentile_filtering=False,percentile=50,radius= 0.01):

    df = df_res.copy()
    #df.reset_index(inplace=True, names="terrain")

    # col = ["slope","slope_std","x_95","y_std_95"]
    

    fig, axs = plt.subplots(3,1,sharex=True)
    fig.set_figwidth(9)
    fig.set_figheight(3)
    fig.subplots_adjust(hspace=0.4,wspace=0.4)

    list_y_coordinates = ["_rotationnal_energy_metric","_translationnal_energy_metric","_total_energy_metric"]
    
    for metric_name,ax,ylabel in zip(list_y_coordinates,axs,list_y_coordinates):
        
        
        for robot in df.robot.unique():
            linestyle_dict = {"husky":"--","warthog":"-"}
            linestyle = linestyle_dict[robot]
            first_time_robot=True
            df_robot = df.loc[df.robot==robot]
            for terrain in np.unique(df_robot.terrain):
                df_terrain = df_robot.loc[df_robot.terrain==terrain]
                x = df_terrain["cmd_metric"+metric_name].to_numpy()
                y = df_terrain[y_column+metric_name].to_numpy()

                #y_95 = np.percentile(y,95)
                #mask = y <= y_95
                #x_masked = x[mask]
                #y_masked = y[mask]
            
                x_masked = x
                y_masked = y
                color_dict = {"asphalt":"pink", "ice":"blue","gravel":"orange","grass":"green","sand":"darkgoldenrod","avide":"grey","avide2":"grey","mud":"darkgoldenrod","tile":"lightcoral"}
                
                


                if percentile_filtering:
                    x_masked,y_masked,y_windows_std = moving_average(x,y,percentile=percentile,r = radius)
                    color_list = [color_dict[terrain]]*x_masked.shape[0]
                    # Scatter plot the data points 
                    if percentile==50:
                        ax.errorbar(x_masked, y_masked,yerr=y_windows_std, color=color_list[0],alpha=0.9,label=terrain)    
                    else:
                        test=1
                       
                        ax.scatter(x_masked, y_masked, color=color_list,alpha=0.9,s=0.8)
                        ax.plot(x_masked, y_masked, color=color_list[0],alpha=1,label=terrain,ls=linestyle)
                    
                    
                    
                else:
                    color_list = [color_dict[terrain]]*x_masked.shape[0]
                    ax.scatter(x_masked, y_masked, color=color_list,alpha=0.2,s=0.8,label=terrain)
                
            x = df_robot["cmd_metric"+metric_name].to_numpy()
            y = df_robot[y_column+metric_name].to_numpy()
            x_masked,y_masked,y_windows_std = moving_average(x,y,percentile=percentile,r = radius)
            ax.plot(x_masked, y_masked, color="black",alpha=1,label=f"{robot} all \n terrain 95%",ls=linestyle)
            
            
        ax.set_ylabel(f"{ylabel[1:]} @(x=95 \%) ")
                
    # Add labels and legend
    axs[2].set_xlabel("Difficulty metric (1-slope)")
    
    axs[0].set_title('New mapping of DRIVE (mass include)'+suffix)
    
    for ax in np.ravel(axs):
        ax.legend()
        ax.grid(True)
        ylimit = ax.get_ylim()
        ax.set_ylim(1,ylimit[1])
        ax.set_xlim(0,1)
        ax.set_yscale("log")

    axs.reshape(3,1)
    
def plot_scatter_with_colormap(x, y, z, ax,cmap='viridis',zlabel=""):
    """
    Plot a scatter plot of x, y with colors based on z values.
    
    Parameters:
    - x: array-like, x-coordinates of the points.
    - y: array-like, y-coordinates of the points.
    - z: array-like, z-values that will determine the color of each point.
    - cmap: string, optional, colormap to use for coloring (default is 'viridis').
    
    Returns:
    - A scatter plot with a color map based on z-values.
    """
    
    # Create a scatter plot
    scatter = ax.scatter(x, y, c=z, cmap=cmap, edgecolors='k', alpha=0.7)
    
    # Add a colorbar to the plot
    plt.colorbar(scatter, label=zlabel)
    
    # Label the axes
    
    # Title of the plot
    #plt.title('Scatter Plot with Z as Color Map')
    
    # Show the plot
    #plt.show()

def plot_scatter_metric(df):
    df_ice = df.loc[df.terrain=="ice"]
    
    fig,axs = plt.subplots(3,3)
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.rotationnal_energy_metric, axs[0,0],cmap='viridis',zlabel="rotationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.translationnal_energy_metric, axs[1,0],cmap='viridis',zlabel="translationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.total_energy_metric,axs[2,0], cmap='viridis',zlabel="total")
    for ax in axs[:,0]:
        ax.set_facecolor("lightcyan")
        
        ax.set_ylabel('Cmd x vel')
    
    df_ice = df.loc[df.terrain=="asphalt"]
    
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.rotationnal_energy_metric, axs[0,1],cmap='viridis',zlabel="rotationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.translationnal_energy_metric, axs[1,1],cmap='viridis',zlabel="translationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.total_energy_metric,axs[2,1], cmap='viridis',zlabel="total")
    
    for ax in axs[:,1]:
        ax.set_facecolor("whitesmoke")

    df_ice = df.loc[df.terrain=="sand"]
    
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.rotationnal_energy_metric, axs[0,2],cmap='viridis',zlabel="rotationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.translationnal_energy_metric, axs[1,2],cmap='viridis',zlabel="translationnal")
    plot_scatter_with_colormap(df_ice.cmd_body_yaw_vel, df_ice.cmd_body_lin_vel, df_ice.total_energy_metric,axs[2,2], cmap='viridis',zlabel="total")
    
    for ax in axs[:,2]:
        ax.set_facecolor("lightyellow")

    for ax in axs[1,:]:
        ax.set_xlabel('Cmd x vel')
def plot_hist(metric_raw,ax,ylabel=""):

    ax.hist(metric_raw,range=(0,1),bins=60,density=True)
    y_lim = ax.get_ylim()
    ax.set_ylabel(ylabel)
    ax.vlines(np.median(metric_raw),ymin=y_lim[0],ymax=y_lim[1],label="median", color="red")
    ax.vlines(np.mean(metric_raw),ymin=y_lim[0],ymax=y_lim[1],label="mean", color="green" )
    ax.legend()
    
def plot_histogramme_metric(df):
    
    df_ice = df.loc[df.terrain=="ice"]
    
    fig,axs = plt.subplots(3,3)
    plot_hist( df_ice.rotationnal_energy_metric, axs[0,0],ylabel="rotationnal")
    plot_hist( df_ice.translationnal_energy_metric, axs[1,0],ylabel="translationnal")
    plot_hist( df_ice.total_energy_metric,axs[2,0], ylabel="total")
    for ax in axs[:,0]:
        ax.set_facecolor("lightcyan")

    df_ice = df.loc[df.terrain=="asphalt"]
    
    plot_hist( df_ice.rotationnal_energy_metric, axs[0,1],ylabel="rotationnal")
    plot_hist( df_ice.translationnal_energy_metric, axs[1,1],ylabel="translationnal")
    plot_hist( df_ice.total_energy_metric,axs[2,1],ylabel="total")
    
    for ax in axs[:,1]:
        ax.set_facecolor("whitesmoke")

    df_ice = df.loc[df.terrain=="sand"]
    
    plot_hist( df_ice.rotationnal_energy_metric, axs[0,2],ylabel="rotationnal")
    plot_hist( df_ice.translationnal_energy_metric, axs[1,2],ylabel="translationnal")
    plot_hist( df_ice.total_energy_metric,axs[2,2],ylabel="total")
    
    for ax in axs[:,2]:
        ax.set_facecolor("lightyellow")

def boxplot(df):

    list_array_transl = []
    list_array_rot = []
    list_array_total = []
    list_terrain = []

    for terrain in list(df.terrain.unique()):
        list_terrain.append(terrain) 
        list_array_total.append(df.total_energy_metric.loc[df["terrain"] == terrain])
        list_array_rot.append(df.rotationnal_energy_metric.loc[df["terrain"] == terrain])
        list_array_transl.append(df.translationnal_energy_metric.loc[df["terrain"] == terrain])
        
    fig, axs = plt.subplots(3,1)
    
    axs[0].boxplot(list_array_rot,showfliers=False,tick_labels=list_terrain)
    axs[0].set_ylabel("rotationnal_energy_metric")
    axs[1].boxplot(list_array_transl,showfliers=False,tick_labels=list_terrain)
    axs[1].set_ylabel("translationnal_energy_metric")
    axs[2].boxplot(list_array_total,showfliers=False,tick_labels=list_terrain)
    axs[2].set_ylabel("total_energy_metric")
    
if __name__ =="__main__":
    
    PATH_TO_RESULT = "drive_datasets/results_multiple_terrain_dataframe/metric/results_slope_metric.csv"
    df = pd.read_csv(PATH_TO_RESULT)
    print(df.columns)
    plot_metric_3d(df,suffix="")
    plt.show()
    plot_metric_depending_on_sampling_space(df)
    plot_metric_scatter(df,alpha_param=0.4,suffix="",show_ellipse=False)
    print(df.head(5))
    plt.show()
    #print(df.std_metric_total_energy_metric)

    #####
    df_res_warthog = pd.read_csv("drive_datasets/results_multiple_terrain_dataframe/metric/warthog_metric_cmd_raw_slope_metric_scatter.csv")
    df_res_husky = pd.read_csv(f"drive_datasets/results_multiple_terrain_dataframe/metric/husky_metric_cmd_raw_slope_metric_scatter.csv")
    df_res = pd.concat([df_res_husky,df_res_warthog],axis=0) 

    print(df_res_husky.terrain.unique())
    plot_metric_scatter_scatter(df_res,alpha_param=0.4,suffix="",percentile_filtering=True,percentile=95,radius= 0.03)#y_column="wheels_metric"), y_column="cmd_diff_icp"
    
    #####
    path_to_raw_result = "drive_datasets/results_multiple_terrain_dataframe/metric/warthog_metric_cmd_raw_slope_metric.csv"
    df_warthog = pd.read_csv(path_to_raw_result)
    
    path_to_raw_result = "drive_datasets/results_multiple_terrain_dataframe/metric/husky_metric_cmd_raw_slope_metric.csv"
    df_husky = pd.read_csv(path_to_raw_result)

    print(df_husky.terrain.unique())
    
    df = pd.concat([df_warthog,df_husky],axis=0)


    #boxplot(df)
    #print(df.columns)
    plot_scatter_metric(df)
    plot_histogramme_metric(df)
    plt.show()


