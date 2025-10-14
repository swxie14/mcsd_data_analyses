"""
created 25.4.16

code to perform pca and sort by option

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve
import utils as u
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def pca_options_mcsd(subjs=[], group='behEphys', z_flag=0):

    # generate smoothing kernel
    kernel_dur = 1000   # ms
    kernel_sigma = 250  # ms
    ax = np.arange(-kernel_dur // 2 + 1., kernel_dur // 2 + 1.)
    kernel = np.exp(-0.5 * (ax / kernel_sigma)**2)
    kernel = kernel / kernel.sum()

    # set downsampling bin
    downsample_bin = 300  # ms

    # define task epoch lengths
    x_state_1 = np.arange(1801)
    x_state_2 = np.arange(1301)
    x_state_3 = np.arange(1201)
    x_outcome = np.arange(1001)
    all_x = [x_state_1, x_state_2, x_state_3, x_outcome]
    num_epochs = len(all_x)
    epoch_len = np.zeros(num_epochs+1)
    for e_idx in range(num_epochs):
        epoch_len[e_idx+1] = (len(all_x[e_idx]) - len(all_x[e_idx]) % downsample_bin) / downsample_bin
    trial_len = int(np.sum(epoch_len))
    epoch_len[1:] = epoch_len[1:]-1
    epoch_len = np.cumsum(epoch_len).astype(int)

    # set home directory
    data_dir = ('C:\\Users\\cooper\\Desktop\\oLab\\mcsd\\data\\' + group + '\\') 

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    for s_idx, subj in enumerate(subjs):

        beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subj, rmv_flag=1)

        # get behavior and task data
        t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1
        if beh_data['tGoalOn'].iloc[-1] > 0:      # find total session time
            t_session = np.max([beh_data['tOutOn'].iloc[-1], beh_data['tGoalOn'].iloc[-1]]) 
        else:
            t_session = beh_data['tOutOn'].iloc[-1] 
        t_session = t_session + 2000 - t_beginning

        num_trials = beh_data.shape[0]
        choice_1 = beh_data['choice1'].values
        choice_2 = beh_data['choice2'].values
        option = choice_1 - 1 + choice_1 - 1 + choice_2 - 1 + 1
        num_options = np.max(option)

        # generate timestamp index mats for relevant task epochs
        state_1_map = np.tile(x_state_1, (num_trials, 1))
        t_tmp = beh_data['tState1On'].values - t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_1_map.shape[1]))
        state_1_map = state_1_map + t_tmp
        state_1_map = state_1_map.astype(int)

        state_2_map = np.tile(x_state_2, (num_trials, 1))
        t_tmp = beh_data['tState2On'].values - t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_2_map.shape[1]))
        state_2_map = state_2_map + t_tmp
        state_2_map = state_2_map.astype(int)

        state_3_map = np.tile(x_state_3, (num_trials, 1))
        t_tmp = beh_data['tState3On'].values - t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_3_map.shape[1]))
        state_3_map = state_3_map + t_tmp
        state_3_map = state_3_map.astype(int)

        outcome_map = np.tile(x_outcome, (num_trials, 1))
        t_tmp = beh_data['tOutOn'].values - t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, outcome_map.shape[1]))
        outcome_map = outcome_map + t_tmp
        outcome_map = outcome_map.astype(int)

        all_maps = [state_1_map, state_2_map, state_3_map, outcome_map]

        # get ephys data
        cell_names = ephys_data.columns
        num_cells = cell_names.shape[0]

        # run through cells
        for c_idx in range(num_cells):
            spike_times = ephys_data[cell_names[c_idx]].values
            spike_times = spike_times[0][0][0] - t_beginning
            spikes = np.zeros(int(t_session))
            spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
            spikes[spike_times.astype(int)] = 1

            spikes = convolve(spikes, kernel, mode='same')      # smooth spikes

            for o_idx in range(num_options):
                for e_idx in range(num_epochs):
                    option_epoch_map = all_maps[e_idx][option==o_idx+1, :]
                    epoch_psth = np.mean(spikes[option_epoch_map], axis=0)

                    trimmed_len = len(epoch_psth) - len(epoch_psth) % downsample_bin
                    trimmed = epoch_psth[:trimmed_len]
                    epoch_psth =  trimmed.reshape(-1, downsample_bin).mean(axis=1)

                    if e_idx == 0:
                        option_psths = epoch_psth
                    else:
                        option_psths = np.concatenate((option_psths, epoch_psth))

                if o_idx == 0:
                    cell_psths = option_psths
                else:
                    cell_psths = np.concatenate((cell_psths, option_psths))

            if c_idx == 0:
                subj_psths = (cell_psths - cell_psths.mean()) / cell_psths.std(ddof=1)          # z-score
                subj_regions = [ephys_data[cell_names[c_idx]].values[0][0][1][0][1:]]
            else:
                cell_psths = (cell_psths - cell_psths.mean()) / cell_psths.std(ddof=1)
                subj_psths = np.vstack((subj_psths, cell_psths))
                subj_regions = np.concatenate((subj_regions, [ephys_data[cell_names[c_idx]].values[0][0][1][0][1:]]))
        
        if s_idx == 0:
            all_psths = subj_psths
            all_regions = subj_regions
        else:
            all_psths = np.vstack((all_psths, subj_psths))
            all_regions = np.concatenate((all_regions, subj_regions))

    regions = np.unique(all_regions)
    colors = plt.cm.cool(np.linspace(0, 1, num_options))

    for r_idx, region in enumerate(regions):
        region_idx = [i for i, text in enumerate(all_regions) if region == text]
        region_psths = all_psths[region_idx, :]
        
        pca = PCA(n_components=3)
        region_pca = pca.fit_transform(region_psths.T)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for o_idx in range(num_options):
            option_idx = np.arange(trial_len*o_idx, trial_len*(o_idx+1)) 
            ax.plot(
                region_pca[option_idx, 0],
                region_pca[option_idx, 1],
                region_pca[option_idx, 2], 
                color=colors[o_idx])
            for e_idx in range(num_epochs+1):
                ax.plot(
                    region_pca[epoch_len[e_idx] + o_idx*(trial_len), 0],
                    region_pca[epoch_len[e_idx] + o_idx*(trial_len), 1],
                    region_pca[epoch_len[e_idx] + o_idx*(trial_len), 2], 
                    'o', markerfacecolor=colors[o_idx],markeredgecolor=colors[o_idx], markersize=5,label='_nolegend_')
            
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.suptitle(f"all subjects: {group}, region: {region}")
        plt.legend(['LL', 'LR', 'RL', 'RR'])
        plt.show(block=False)

        # # Function to update view angle
        # def update(frame):
        #     ax.view_init(elev=30, azim=frame)

        # # Create animation
        # ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

        # # Save as MP4
        # ani.save("C:\\Users\\cooper\\Desktop\\oLab\\mcsd\\presentations\\25.6.9 lab meeting\\pca_option_sma.gif", fps=20, dpi=200, bitrate=1800)

if __name__ == "__main__":
    group='behEphys'
    z_flag=0
    subjs = ['subj_057', 'subj_058', 'subj_059', 'subj_060', 'subj_062', 'subj_063', 'subj_064', 'subj_065', 'subj_066', 'subj_067']
    result = pca_options_mcsd(subjs=[], group=group, z_flag=z_flag)