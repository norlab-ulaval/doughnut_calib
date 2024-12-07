# Import
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

FILE_PATH = "tests_figures/mean_heat_map_gma_warthog_metric.csv"
SQUARES_TO_ANALYZE = [{'x': -0.5, 'y':4, 'width': 1, 'height': 1, 'axis': 'linear'},
                      {'x': 3.5, 'y':-1, 'width': 0.5, 'height': 2, 'axis': 'yaw'},]
COLORMAP_TERRAIN = {"asphalt":"grey", "ice":"blue","gravel":"orange","grass":"green","sand":"orangered","avide":"grey","avide2":"grey","mud":"darkgoldenrod","tile":"lightcoral"}
# Open the csv file as a dataframe
data = pd.read_csv(FILE_PATH)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
plt.rc('font', **font)
plot_fs = 12
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('axes', labelsize=10)
mpl.rcParams['lines.dashed_pattern'] = [2, 2]
mpl.rcParams['lines.linewidth'] = 1.0 


# Find every unique value in the 'terrain' column
terrains = data['terrain'].unique()

columns_to_analyze = ['first_window_metric', 'last_window_metric', 'last_window_cmd_total_energy_metric']

terrain_list = []
last_window_metric = []
last_window_cmd_total_energy_metric = []
list_window_metric_std = []
list_window_cmd_total_energy_metric_std = []
for terrain in terrains:
    #if terrain == 'gravel' or terrain == 'asphalt':
    #    continue
    # Find the rows that have the terrain value
    terrain_data = data[data['terrain'] == terrain]
    for square in SQUARES_TO_ANALYZE:
        square_data = terrain_data[(abs(terrain_data['cmd_body_yaw_mean']) >= square['x']) & (abs(terrain_data['cmd_body_yaw_mean']) <= square['x'] + square['width']) & (abs(terrain_data['cmd_body_x_mean']) >= square['y']) & (abs(terrain_data['cmd_body_x_mean']) <= square['y'] + square['height'])]
        terrain_list.append(terrain)
        for column in columns_to_analyze:
            print(f"Analyzing {column} for terrain {terrain} in square {square}")
            # Find the mean of the square
            median = square_data[column].median()
            print(f"Mean: {median}")
            std = square_data[column].std()
            print(f"STD: {std}")
            if column == 'last_window_metric':
                last_window_metric.append(median)
                list_window_metric_std.append(std)
            elif column == 'last_window_cmd_total_energy_metric':
                last_window_cmd_total_energy_metric.append(median)
                list_window_cmd_total_energy_metric_std.append(std)
                #if square['axis'] == 'linear':
                #    last_window_cmd_total_energy_metric.append(square_data['last_window_cmd_translationnal_energy_metric'].median())
                #    list_window_cmd_total_energy_metric_std.append(square_data['last_window_cmd_translationnal_energy_metric'].std())
                #elif square['axis'] == 'yaw':
                #    last_window_cmd_total_energy_metric.append(square_data['last_window_cmd_rotationnal_energy_metric'].median())
                #    list_window_cmd_total_energy_metric_std.append(square_data['last_window_cmd_rotationnal_energy_metric'].std())

# Plot the results with last_window_metric as x and last_window_cmd_total_energy_metric as y
for i in range(len(terrain_list)):
    plt.scatter(last_window_metric[i], last_window_cmd_total_energy_metric[i], color=COLORMAP_TERRAIN[terrain_list[i]], label=terrain_list[i])
    plt.errorbar(last_window_metric[i], last_window_cmd_total_energy_metric[i], xerr=2*list_window_metric_std[i], yerr=2*list_window_cmd_total_energy_metric_std[i], fmt='o', color=COLORMAP_TERRAIN[terrain_list[i]])
# Build a custom legend at the bottom of the plot for the terrains
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=terrain[0].upper() + terrain[1:], markerfacecolor=COLORMAP_TERRAIN[terrain], markersize=10) for terrain in terrains]
plt.legend(handles=legend_elements, loc='upper right')
plt.xlabel('Metric')
plt.ylabel('Danger degree (J)')
fig = plt.gcf()
fig.set_figwidth(88*2/25.4)
plt.show()
plt.savefig('tests_figures/last_window_metric_vs_last_window_cmd_total_energy_metric.pdf', format='pdf')