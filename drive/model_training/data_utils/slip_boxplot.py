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
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('axes', labelsize=10)
    mpl.rcParams['lines.dashed_pattern'] = [2, 2]
    mpl.rcParams['lines.linewidth'] = 1.0

    fig, axs = plt.subplots(3,1)
    fig.set_figwidth = 88/25.4
    fig.set_figheight = 4.58
    
    fig.subplots_adjust(hspace=0.2 ,wspace=0.4)
    
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
            #pos_labels += (value/2 * delta_same_terrain) 
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
    iterator = 0
    for box, color_list in zip([box1,box2,box3], [list_color_x, list_color_y, list_color_rot]):
        
        for patch, color in zip(box['boxes'], color_list):

            patch.set_facecolor(color)  # Change to your desired color
            patch.set_alpha(alpha_bp)
        # Change the median line color to black
        for median in box['medians']:
            median.set_color('black')
    #list_terrain_x_ticks = [terrain[0].capitalize() + terrain[1:] for terrain in list_terrain]
    #axs[0].set_xticks(list_pos_labels,labels=[])
    #axs[1].set_xticks(list_pos_labels,labels=[])
    #axs[2].set_xticks(list_pos_labels,labels=list_terrain_x_ticks)

    for ax in np.ravel(axs):
        ax.set_xticks([])       # Remove the ticks
        ax.set_xticklabels([])  # Remove the labels
        
    axs[0].set_ylabel("Longitudinal slip [m/s]")
    axs[1].set_ylabel("Lateral slip [m/s]")
    axs[2].set_ylabel("Angular slip [rad/s]")
    tick_labels = ['Asphalt', 'Grass', 'Gravel', 'Sand', 'Ice', 'Overall']
    ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    axs[2].set_xticks(ticks, tick_labels)

    # Extract legends from both axes
    #legend1 = axs[0].get_legend_handles_labels()
    # Combine legends from both axes
    #handles = legend1[0] 
    #labels = legend1[1]

    #print(labels)
    #final_handles = [handles[4],handles[5]]
    #final_labels = [labels[4][0].capitalize() + labels[4][1:],labels[5][0].capitalize() + labels[5][1:]]
    
    #axs[0].legend(handles=final_handles,labels=final_labels)
    #axs[1].set_ylabel("translationnal_energy_metric")
    #axs[2].set_ylabel("total_energy_metric")

    for ax in np.ravel(axs):
        ax.set_xlim(delta_x/2,list_pos_hfill[-1])
        # ax.set_ylim(0,1)
    axs[0].set_ylim(-0.1, 2.1)
    axs[1].set_ylim(-0.1, 2.1)
    axs[2].set_ylim(-0.1, 6)

    ## Add the color fill 
    j = 1 
    print(list_position)
    print(list_pos_hfill)

    print(list_terrain)
    # for color_x,color_y,color_rot,label  in zip(list_color_x,list_color_y,list_color_rot,list_terrain_reordered_rot):
        
    #     axs[0].fill_between(list_pos_hfill[j-1:j+1],y1=-6, y2=6,color="lightgrey",alpha=alpha_param)
    #     axs[1].fill_between(list_pos_hfill[j-1:j+1],y1=-6 , y2=6,color="lightgrey",alpha=alpha_param)
    #     axs[2].fill_between(list_pos_hfill[j-1:j+1],y1=-6, y2=6,color="lightgrey",alpha=alpha_param,label=label[0].upper()+label[1:])
        
    #     j+=2
    
    # Add the vertical thick line 
    # axs[0].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",linewidth=linewidth_overall)
    # axs[1].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",linewidth=linewidth_overall)
    # axs[2].vlines(list_pos_hfill[-3],ymax=10,ymin=-10,color="black",linewidth=linewidth_overall)
    #axs[1].fill_between(list_pos_hfill[j-1:j+1],y1=1,color=color_transl,alpha=alpha_param)
    #axs[2].fill_between(list_pos_hfill[j-1:j+1],y1=1,color=color_total,alpha=alpha_param,label=label[0].upper()+label[1:])
    #    
    # handles, labels = axs[2].get_legend_handles_labels()
    # print(handles)
    # print(labels)
    # overall = mpatches.Patch(edgecolor='black',facecolor="white")
    # handles[-1] = overall
    # fig.legend(handles,labels,bbox_to_anchor= (0.78,0.125),ncols=3)
    #fig.tight_layout()

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