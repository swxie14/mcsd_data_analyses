"""
created 25.4.4

load behavior and ephys matlab data

"""

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_ephys_data_mcsd(data_dir='/Users/samuelxie/Desktop/oLab/behEphys/',subj=[],rmv_flag=1):

    if not subj:
        raise ValueError("subject must be provided")
    
    # import .mat files and convert matlab struct to pandas dataframe
    beh_data = loadmat(data_dir + subj + '/' + subj + '_mcsdEphysNlx.mat')
    beh_data = beh_data['tS']
    beh_data_dict = {}
    for field_name in beh_data.dtype.names:
        values = beh_data[field_name].flatten()
        beh_data_dict[field_name] = [v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]
    beh_data = pd.DataFrame.from_dict(beh_data_dict)

    ephys_data = loadmat(data_dir + subj + '/' + subj + '_mcsdSpikes.mat')
    ephys_data = ephys_data['sS']
    ephys_data_dict = {}
    for field_name in ephys_data.dtype.names:
        ephys_data_dict[field_name] = ephys_data[field_name].flatten()
    ephys_data = pd.DataFrame.from_dict(ephys_data_dict)

    if rmv_flag:
        # find empty rows in the beh_data choice2 column and remove those rows from the beh_data
        no_response = beh_data['choice2'].apply(lambda x: isinstance(x, np.ndarray) and x.size == 0)
        beh_data = beh_data[~no_response]
    return beh_data, ephys_data


def load_behavior_mdl_mcsd(behavior_mdl=[]):

    if not behavior_mdl:
        raise ValueError("model must be provided")
    
    data_dir='/Users/samuelxie/Desktop/oLab/'

    # import .mat files and convert matlab struct to pandas dataframe
    mdl_data = loadmat(data_dir + behavior_mdl + '_behEphys.mat')
    mdl_data = mdl_data['regStruct']
    mdl_data_dict = {}
    for field_name in mdl_data.dtype.names:
        #values = mdl_data[field_name].flatten()
        #mdl_data_dict[field_name] = [v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]
        # Extract the array data from the struct field
        field_data = mdl_data[field_name][0, 0]  # Get data from the scalar struct
        
        # Flatten and clean the data
        if field_name == 'subjId':
            # Special handling for subjId which contains nested arrays
            clean_values = [str(item[0]) if isinstance(item, np.ndarray) and len(item) > 0 else str(item) for item in field_data.flatten()]
        else:
            # For numerical fields, extract the values
            clean_values = [item[0] if isinstance(item, np.ndarray) and len(item) > 0 else item for item in field_data.flatten()]
        
        mdl_data_dict[field_name] = clean_values
    
    mdl_data = pd.DataFrame.from_dict(mdl_data_dict)

    return mdl_data


def time_btwn_state_1_resp_1(behEphys='behEphys', subjs=[]):

    data_dir = ('/Users/samuelxie/Desktop/oLab/' + behEphys + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)
    
    time_arr= []

    for s_idx in range(n_subjs):
        beh_data, ephys_data = load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        state_1 = beh_data['tState1On'].values
        resp_1 = beh_data['tResp1'].values
        time_btwn_state_1_resp_1 = resp_1 - state_1
        for time in time_btwn_state_1_resp_1:
            time_arr.append(float(time))  # Convert to regular Python float
    
    print(time_arr)
    print("mean: ", np.mean(time_arr))
    print("median: ", np.median(time_arr))
    print("std: ", np.std(time_arr))


def time_btwn_fix_on_resp_1(behEphys='behEphys', subjs=[]):

    data_dir = ('/Users/samuelxie/Desktop/oLab/' + behEphys + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)
    
    time_arr= []

    for s_idx in range(n_subjs):
        beh_data, ephys_data = load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        fix = beh_data['tFix'].values
        resp_1 = beh_data['tResp1'].values
        time_btwn_fix_on_resp_1 = resp_1 - fix
        for time in time_btwn_fix_on_resp_1:
            time_arr.append(float(time))  # Convert to regular Python float
    
    print(time_arr)
    print("mean: ", np.mean(time_arr))
    print("median: ", np.median(time_arr))
    print("std: ", np.std(time_arr))


def time_btwn_fix_on_state_1(behEphys='behEphys', subjs=[]):

    data_dir = ('/Users/samuelxie/Desktop/oLab/' + behEphys + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)
    
    time_arr= []

    for s_idx in range(n_subjs):
        beh_data, ephys_data = load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        fix = beh_data['tFix'].values
        state_1 = beh_data['tState1On'].values
        time_btwn_fix_on_state_1 = state_1 - fix
        for time in time_btwn_fix_on_state_1:
            time_arr.append(float(time))  # Convert to regular Python float
    
    #print(time_arr)
    print("mean: ", np.mean(time_arr))
    print("median: ", np.median(time_arr))
    print("std: ", np.std(time_arr))
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(time_arr, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(time_arr), color='red', linestyle='--', label=f'Mean: {np.mean(time_arr):.1f}')
    plt.axvline(np.median(time_arr), color='green', linestyle='--', label=f'Median: {np.median(time_arr):.1f}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Between Fix On and State 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=True)

if __name__ == "__main__":
    #time_btwn_state_1_resp_1()
    time_btwn_fix_on_state_1()

