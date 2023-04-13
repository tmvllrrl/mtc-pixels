# Space time plot (stop and take control) -- reference?
def plot_std(pos_vel_collector, n_cars = 22, horizon = 5010, warmup = 2500, results_name=None):
    """
    TODO: Levels = None, write levels at end of warmup and end of trial to a file 
    This solution is still quick and dirty
    """
    import os
    import time 
    import matplotlib.pyplot as plt 
    import matplotlib.colors as colors
    from matplotlib.collections import LineCollection
    import seaborn as sns 
    import numpy as np
    sns.set_style("whitegrid")

    name = str(time.time()).split('.')[0][-6:-1] # Name will update every second
    pos_vel_collector = np.array(pos_vel_collector)
    #print("shape", pos_vel_collector.shape)
    print(f"STD name:{name}\n")

    N_CARS = n_cars
    WARMUP = int(warmup / 10) 
    HORIZON = int(horizon / 10)
    DIR = f"../../michael_files/{results_name}/"

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    path = f"./{DIR}/" + name + ".png"

    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    #print(pos_vel_collector.shape) # (751, 22, 2) # We are writing data every 10 timesteps

    ys = []
    yv = []

    for i in range(N_CARS):
        temp = []
        temp_v = []
        for  j in pos_vel_collector[:,i]:
            temp.append(j[0])
            temp_v.append(j[1])
        ys.append(np.array(temp))
        yv.append(np.array(temp_v))

    c_max = 8.0
    norm = plt.Normalize(0, c_max)
    x = np.arange(WARMUP + HORIZON)

    line_segments = LineCollection([np.column_stack([x, y]) for y in ys],
                               cmap = my_cmap,
                               norm = norm)
                               
    line_segments.set_array(np.arange(np.max(yv) + 1))
    line_segments.set_linewidth(1)

    
    x_and_position = [np.column_stack([x, y]) for y in ys]
    x_and_velocity = [np.column_stack([x, y]) for y in yv]

    #print(len(x_and_position))
    #print(x_and_position[0].shape)

    #All velocities at warmup time 
    all_velocities = []
    for i in range(N_CARS):
        all_velocities.append(x_and_velocity[i][:,1][(WARMUP)]) # -1 not necessary because 251st i.e accessed by 250 is the correct one

    all_velocities = np.array(all_velocities)
    # All velocities min and max
    av_min = round(np.min(all_velocities),2)
    av_max = round(np.max(all_velocities),2)

    fig, ax = plt.subplots(figsize=(16,9), dpi = 300)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(ys), np.max(ys))
    #print(x_and_velocity[0][:,1])
    for i in range(n_cars):
        ax.scatter(x, x_and_position[i][:,1], c = x_and_velocity[i][:,1], alpha = 0.99, marker = '|', cmap = my_cmap)
        
    plt.axvline(x=WARMUP, ls='--', color = 'black', linewidth = '1.5', label = "End of Warmup")
    # plt.axvline(x=5000/10, ls='--', color = 'blue', linewidth = '1.5', label = "Start of interruptions")
    # plt.axvline(x=8000/10, ls='--', color = 'black', linewidth = '1.5', label = "End of interruptions")

    axcb = fig.colorbar(line_segments, norm = norm, )
    axcb.set_label('Velocity (m/s)', fontsize=18)
    axcb.ax.tick_params(labelsize=18)

    # ax.set_title(f"Space-Time diagram: At warmup, max vel. = {av_max} m/s, min vel. ={av_min} m/s", fontsize=18)
    # ax.set_title(f"Space-Time diagram", fontsize=18)
    ax.set_xlabel('Time (s)', fontsize = 18)
    ax.set_ylabel('Position (m)', fontsize = 18)
    ax.set_xticks(np.arange(0,WARMUP + HORIZON, 100))
    ax.legend(fontsize= 16, loc = 'lower right')
    #ax.autoscale()
    plt.savefig(path)
    plt.close()

    return 0 

#plot space time diagram
# plot_std(pos_vel_collector, n_cars = 22, horizon = 5010, warmup = 2500)