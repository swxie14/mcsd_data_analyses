import utils as u
import numpy as np

import statsmodels.api as sm


class Spikes:
    """
    Handles electrophysiological spike data for a single subject.
    
    This class processes spike times from multiple neurons, including creating
    binary spike trains, epoching activity around task events, detrending,
    and z-scoring firing rates.
    
    Attributes:
        ephys_data (pd.DataFrame): DataFrame with columns for each recorded cell.
            Each cell contains spike times and brain region information.
        t_beginning (int): Session start time in milliseconds
        t_session (int): Total session duration in milliseconds
        cell_names (pd.Index): List of all cell identifiers
        num_cells (int): Total number of recorded neurons
        num_trials (int): Number of behavioral trials
    
    Example:
        >>> subj_ephys = Spikes('subj_057', t_beginning=0, t_session=300000, num_trials=120)
        >>> state_map = subj_beh.construct_timestep_index_maps('tState1On', 1000, 200)
        >>> z_scores = subj_ephys.z_score_cell(state_map, 'Cell_001')
    """
    def __init__(self, subj : str, t_beginning:int, t_session : int, num_trials : int):
        _, self.ephys_data = u.load_ephys_beh_data_mcsd(subj)
        self.t_session = t_session
        self.t_beginning = t_beginning
        self.cell_names = self.ephys_data.columns
        self.num_cells = self.cell_names.shape[0]
        self.num_trials = num_trials


    def process_spikes(self, cell_name: str):
        spike_times = self.ephys_data[cell_name].values[0] - self.t_beginning
        spikes_cell = np.zeros(int(self.t_session))
        spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < self.t_session)]
        spikes_cell[spike_times.astype(int)] = 1
        return spikes_cell
    

    def z_score_cell(self, state_map : np.ndarray, cell_name:str, fixed=True):
        cell_spikes = self.process_spikes(cell_name)
        if fixed:
            epoch_spikes = np.sum((cell_spikes[state_map]), axis = 1) #n x n
        else:
            epoch_spikes = np.zeros(self.num_trials)
            # Sum spikes during state 1 period for each trial
            for trial in range(self.num_trials):
                if state_map[trial].size > 0:  # Check if there are time points
                    epoch_spikes[trial] = np.sum(cell_spikes[state_map[trial]]) / len(state_map[trial])
                else:
                    epoch_spikes[trial] = 0 
                    
        spikes = None
        if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5:
            #filter any correlation between spikes and trial number
            X = np.array(range(self.num_trials)) + 1
            X = sm.add_constant(X)   # (100, 2); X[i] = [1, i]
            model = sm.GLM(epoch_spikes, X, family=sm.families.Poisson()).fit()
            if model.pvalues[1] < 0.05:
                epoch_spikes = epoch_spikes - model.fittedvalues
            
            spikes = (epoch_spikes - np.mean(epoch_spikes))/np.std(epoch_spikes)
            
        return spikes


    def get_region(self, cell_name : str, full_region=True):
        if not full_region:
            return self.ephys_data[cell_name][1][1:]
        return self.ephys_data[cell_name][1]
        
        