import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.patches as mpatches
import sys
import os
project_root = os.path.abspath("/home/william/workspaces/drive_ws/src/DRIVE/")
if project_root not in sys.path:
    sys.path.append(project_root)
from extractors import *
    
def reorder_boxplot(list_array, list_color,list_terrain):

    # compute the list of median 
    list_median = []
    for terrain in list_array:
        list_median.append(terrain.median())
    # List median 
    np_array = np.array(list_median)

    # Order to applied
    order = np.argsort(np_array)
    
    list_data = []
    list_color_oredered = []
    list_terrain_reoredered = []
    for order_i in order:
        list_data.append(list_array[order_i])
        list_color_oredered.append(list_color[order_i])
        list_terrain_reoredered.append(list_terrain[order_i])

    return list_data, list_color_oredered, list_terrain_reoredered

def boxplot_all_terrain_warthog_robot(df,alpha_param=0.3,robot="warthog", 
                                    alpha_bp=0.4,path_to_save="figure/fig_slip_boxplot.pdf",
                                    linewidth_overall = 5):

    df = df.loc[df.robot == "warthog"]

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=10)
    mpl.rcParams['lines.dashed_pattern'] = [2, 2]
    mpl.rcParams['lines.linewidth'] = 1.0

    fig = plt.figure(figsize=(88/25.4, 4.58))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    axs = [ax1, ax2, ax3]
    # fig.set_figwidth = 88/25.4
    # fig.set_figheight = 4.58
    
    fig.subplots_adjust(left=0.2)
    
    list_array_x = []
    list_array_y = []
    list_array_rot = []
    list_terrain = []
    list_robot = []
    list_robot_name = []
    
    dico_robot_name = {"husky":"orange","warthog": "grey","Overall":"blue"}
    color_dict = {"asphalt":"grey", "ice":"blue","gravel":"#cda66a",
                "grass":"green","sand":"orangered","avide":"grey",
                "avide2":"grey","mud":"darkgoldenrod","tile":"lightcoral",
                "Overall":"white"}
    
    # Compute and reorder the boxplot 
    for terrain in list(df.terrain.unique()):
        df_terrain = df.loc[df.terrain==terrain]
        nb_robot = 0
        if terrain == "tile":
            continue
        else:
            
            nb_robot += 1 
            df_robot = df_terrain.loc[df_terrain.robot==robot]
            
            
            list_array_x.append(np.abs(df_robot.slip_body_x_ss.loc[df_robot["terrain"] == terrain]))
            list_array_y.append(np.abs(df_robot.slip_body_y_ss.loc[df_robot["terrain"] == terrain]))
            list_array_rot.append(np.abs(df_robot.slip_body_yaw_ss.loc[df_robot["terrain"] == terrain]))

            list_robot_name.append(robot)
        list_robot.append(nb_robot)
        list_terrain.append(terrain)
    list_robot_name.append(robot)
    
    list_color = [color_dict[terrain] for terrain in df.terrain.unique()]

    list_array_x, list_color_x,list_terrain_reordered_x = reorder_boxplot(list_array_x, list_color,list_terrain )
    list_array_y, list_color_y,list_terrain_reordered_y = reorder_boxplot(list_array_y, list_color,list_terrain)
    list_array_rot, list_color_rot,list_terrain_reordered_rot = reorder_boxplot(list_array_rot, list_color,list_terrain)
    
    for _list in [list_array_rot, list_color_rot, list_terrain_reordered_rot]:
        tmp = _list[-2]
        _list[-2] = _list[-1]
        _list[-1] = tmp
    # 
    # Add the overall 
    list_array_x.append([item for sublist in list_array_x for item in sublist])
    list_array_y.append([item for sublist in list_array_y for item in sublist])
    list_array_rot.append([item for sublist in list_array_rot for item in sublist])
    list_terrain.append("Overall")
    list_color_x.append("white")
    list_color_y.append("white")
    list_color_rot.append("white")
    list_terrain_reordered_rot.append("Overall")
    list_robot.append(1)
    # Compute the position 
    delta_x = 0.5
    delta_same_terrain = 0.35
    position = 0
    list_position = []
    box_width = 0.3
    list_pos_labels = []
    list_pos_hfill = []
    # Compute the pos of boxes and pose of terrain labels
    pos_labels = 0
    pos_hfill = 0
    for value in list_robot:
        list_pos_hfill.append(pos_hfill)
        if value != 1:
            for i in range(1,value+1,1):
                if i ==1:
                    position += delta_x 
                    list_position.append(position)
                else:
                    position += delta_same_terrain 
                    list_position.append(position)
            pos_labels += delta_x + (delta_same_terrain/2) 
            list_pos_labels.append(pos_labels)
            pos_labels += (delta_same_terrain/2)
        else:
            position += delta_x
            pos_labels += delta_x
            list_position.append(position)
            list_pos_labels.append(pos_labels)
            
        pos_hfill = position + delta_x/2
        list_pos_hfill.append(pos_hfill)

    box1 = axs[0].boxplot(list_array_x,whis=(2.5, 97.5),showfliers=False,patch_artist=True,positions=list_position,widths=box_width)
    box2 = axs[1].boxplot(list_array_y,whis=(2.5, 97.5),showfliers=False,patch_artist=True,positions=list_position,widths=box_width)
    box3 = axs[2].boxplot(list_array_rot,whis=(2.5, 97.5),showfliers=False,patch_artist=True, positions=list_position,widths=box_width)

    for box, color_list in zip([box1,box2,box3], [list_color_x, list_color_y, list_color_rot]):
        
        for patch, color in zip(box['boxes'], color_list):

            patch.set_facecolor(color)  # Change to your desired color
            patch.set_alpha(alpha_bp)
        # Change the median line color to black
        for median in box['medians']:
            median.set_color('black')

    for ax in np.ravel(axs):
        ax.set_xticks([])       # Remove the ticks
        ax.set_xticklabels([])  # Remove the labels
        
    axs[0].set_ylabel("Longitudinal slip \n (m/s)")
    axs[1].set_ylabel("Lateral slip \n (m/s)")
    axs[2].set_ylabel("Angular slip (rad/s)")
    tick_labels = ['Asphalt', 'Grass', 'Gravel', 'Sand', 'Ice', 'Overall']
    ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    axs[2].set_xticks(ticks, tick_labels)

    for ax in np.ravel(axs):
        ax.set_xlim(delta_x/2,list_pos_hfill[-1])
        # ax.set_ylim(0,1)
    axs[0].set_ylim(-0.1, 2.1)
    axs[1].set_ylim(-0.1, 2.1)
    axs[2].set_ylim(-0.1, 6)
    
    # Add the vertical thick line 
    axs[0].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",alpha=0.5, linewidth=0.75,linestyles="--")
    axs[1].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",alpha=0.5, linewidth=0.75,linestyles="--")
    axs[2].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",alpha=0.5, linewidth=0.75,linestyles="--")

    fig.savefig(path_to_save,dpi=300)
    fig.savefig(path_to_save[:-4]+".png",dpi=300)
    
if __name__ =="__main__":
    
    path_to_raw_result = "drive_datasets/results_multiple_terrain_dataframe/filtered_cleared_path_warthog_following_robot_param_all_terrain_steady_state_dataset.pkl"
    df_warthog = pd.read_pickle(path_to_raw_result)
    # path_to_raw_result = "drive_datasets/results_multiple_terrain_dataframe/metric/husky_metric_cmd_raw_slope_metric.csv"
    # df_husky = pd.read_csv(path_to_raw_result)
    #df_husky = df_husky.drop()
    # df = pd.concat([df_warthog,df_husky],axis=0)

    print_column_unique_column(df_warthog)
    #boxplot(df)
    boxplot_all_terrain_warthog_robot(df_warthog)
    #boxplot_few_robot_few_terrain(df)
    #print(df.columns)
    #plot_scatter_metric(df)
    #plot_histogramme_metric(df)
    plt.show()



    print(0.40732918650830996)
    print(0.75 / 0.40732918650830996)