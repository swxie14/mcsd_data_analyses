import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio



def load_ephys_beh_data_mcsd(subj=None,  rmv_flag = 1):
    """
    Loads subjects electrophysiological data and behavorial data

    Args:
        subj: subject of interest
        rmv_flag: indicates whether rows with empty choice2 values should be removed in the behavorial dataset

    Returns:
        Electrophysiological and behavorial data for each patient

    Example behavioral data structure (beh_data DataFrame):
        - tFix: fix state time
        - currSubgoal: current subgoal value
        - goalAmt: goal amount
        - etc.
        Shape: (n_trials, n_features)
    
    Example Electrophysiological data structure (ephys_data DataFrame):
            1. Cell names (field names):
                ephys_data.dtype.names = ['Cell_001', 'Cell_002', ..., 'Cell_N']
                Each field name represents a single recorded neuron/cell
            
            2. Index Cell Names for Cell-specific data:
                ephys_data['Cell_001'] contains a nested array with:
                - [0]: Spike timestamps - array of shape (1, n_spikes)
                    Contains Unix timestamps (e.g., 1.68112410e+12) for each spike event
                - [1]: Brain region label - string (e.g. 'ACC', 'LH')
                    Anatomical location where the cell was recorded
                - Brain regions without L or R: A, ACC, CM, H, IFG, OF, PRV, PT, SMA
    """
    if subj is None:
        raise ValueError("subject must be provided")

    current_dir = Path.cwd()
    subj_beh = current_dir / 'behEphys' / subj / f'{subj}_mcsdEphysNlx.mat'
    beh_data = sio.loadmat(subj_beh)
    beh_data = beh_data['tS']

    beh_data_dict = {}
    for field_name in beh_data.dtype.names:
        values = beh_data[field_name].flatten()
        #if numpy array has one element treat it as individual element
        beh_data_dict[field_name] = [v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]

    beh_data = pd.DataFrame.from_dict(beh_data_dict)

    if rmv_flag:
        # find empty rows in the beh_data choice2 column and remove those rows from the beh_data
        no_response = beh_data['choice2'].apply(lambda x: isinstance(x, np.ndarray) and x.size == 0)
        beh_data = beh_data[~no_response]


    #load ephys_data
    subj_ephys = current_dir / 'behEphys' / subj / f'{subj}_mcsdSpikes.mat'
    ephys_data = sio.loadmat(subj_ephys)
    ephys_data = ephys_data['sS']

    ephys_data_dict = {}

    for field_name in ephys_data.dtype.names:
        ephys_data_dict[field_name] = ephys_data[field_name].flatten()
        ephys_data_dict[field_name] = [ephys_data_dict[field_name][0][0][0][0], ephys_data_dict[field_name][0][0][1][0]]
    ephys_data = pd.DataFrame.from_dict(ephys_data_dict)



    return beh_data, ephys_data



def load_behavior_mdl_mcsd(behavior_mdl=None):
    """
    Loads subjects model data based on behavior

    Args:
        behavior_mdl: behavior model name

    Returns:
        Electrophysiological and behavorial data for each patient

    Example behavorial model data structure (mdl_data DataFrame):
        - Q_1: option value
        - U_1: option value with exploration biases
        - Qc_1: value of first chosen option
        - etc.
        Shape: (n_trials_subj * subj, n_features)

    """

    if behavior_mdl is None:
        raise ValueError("model must be provided")

    current_dir = Path.cwd()

    mdl_path = current_dir / 'behEphys' / f'{behavior_mdl}_behEphys.mat'
    mdl_data = sio.loadmat(mdl_path)
    mdl_data = mdl_data['regStruct']
    mdl_data_dict = {}

    for field_name in mdl_data.dtype.names:
        field_data = mdl_data[field_name][0, 0]
         # Flatten and clean the data
        if field_name == 'subjId':
            # Special handling for subjId which contains nested arrays
            clean_values = [str(item[0]) if isinstance(item, np.ndarray) and len(item) > 0 else str(item) for item in field_data.flatten()]
        else:
            # For numerical fields, extract the values
            clean_values = [item[0] if isinstance(item, np.ndarray) and len(item) > 0 else item for item in field_data.flatten()]
            #print(np.isnan(clean_values).any())

        mdl_data_dict[field_name] = clean_values

    mdl_data = pd.DataFrame.from_dict(mdl_data_dict)
    return mdl_data

if __name__ == '__main__':
    behavior_mdl='det_hmbOforgetFixedEtaBetaExplorBias'
    #load_behavior_mdl_mcsd(behavior_mdl)
    load_ephys_beh_data_mcsd('subj_057')
    


        
        
        