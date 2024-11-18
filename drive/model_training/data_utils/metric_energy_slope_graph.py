import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 






from matplotlib.patches import Ellipse



def plot_metric_scatter(df_res,alpha_param=0.4,suffix=""):

    df = df_res.copy()
    #df.reset_index(inplace=True, names="terrain")

    # col = ["slope","slope_std","x_95","y_std_95"]
    x = df.metric_total_energy_metric.to_numpy()
    y = df.cmd_95_total_energy_metric.to_numpy() * df.mean_slope_total_energy_metric.to_numpy() 

    
    
    fig, ax = plt.subplots(1,1)
    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.subplots_adjust(hspace=0.4,wspace=0.4)

    color_dict = {"asphalt":"pink", "ice":"blue","gravel":"orange","grass":"green","sand":"darkgoldenrod"}
    
    color_list = [color_dict[terrain] for terrain in df.terrain]
    # Scatter plot the data points 
    ax.scatter(x, y, color=color_list, marker="+")

    # Plot uncertainty ellipses around each data point
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
        plt.gca().add_patch(ellipse)


    # Add labels and legend
    ax.set_xlabel("Difficulty metric (1-slope)")
    ax.set_ylabel("Total ICP Energy values @(x=95 \%) ")
    ax.set_title('New mapping of DRIVE (mass include)'+suffix)
    ax.legend()
    ax.grid(True)
    ylimit = ax.get_ylim()
    ax.set_ylim(0,ylimit[1])
    ax.set_xlim(0,1)


if __name__ =="__main__":
    
    PATH_TO_RESULT = "drive_datasets/results_multiple_terrain_dataframe/metric/results_slope_metric.csv"
    df = pd.read_csv(PATH_TO_RESULT)
    print(df.columns)

    plot_metric_scatter(df,alpha_param=0.4,suffix="")
    plt.show()
    print(df.std_metric_total_energy_metric)