import utils as u
import numpy as np


class Behavior:
    """
    Handles behavioral data for a single subject in the MCSD task.
    
    This class loads and processes behavioral data including trial timings,
    choices, and task events. It provides methods to extract features and
    construct time-aligned index maps for epoching neural data.
    
    Attributes:
        beh_data (pd.DataFrame): Behavioral data with columns for task events
            (e.g., 'tFix', 'tState1On', 'choice1', 'choice2', etc.)
        num_trials (int): Total number of trials completed by the subject
        t_beginning (int): Session start time in milliseconds (with 1000ms buffer)
        t_session (int): Total session duration in milliseconds
    
    Example:
        >>> subj_beh = Behavior('subj_057')
        >>> choices = subj_beh.get_feature_values('choice2')
        >>> state_map = subj_beh.construct_timestep_index_maps('tState1On', 1000, 200)
    """
    def __init__(self, subj : str):
        self.beh_data, _ = u.load_ephys_beh_data_mcsd(subj)
        self.num_trials = self.beh_data.shape[0]
        self.t_beginning = (self.beh_data['tFix'].iloc[0] - 1000) + 1 #creates buffer of 1000 to account for beginning
        self.t_session = self.beh_data['tOutOn'].iloc[-1]  + 2000 - self.t_beginning #creates buffer for ending


    def get_feature_values(self, feature: str):
        return self.beh_data[feature].values

    def construct_timestep_index_maps(self, start : str, window : int, offset=0):
        x_state = np.arange(0, window) + offset 
        state_map = np.tile(x_state, (self.num_trials, 1))
        t_tmp = self.beh_data[start].values - self.t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_map.shape[1]))
        state_map = state_map + t_tmp
        state_map = state_map.astype(int)
        return state_map
    
    def construct_nonfixed_timestep_index_maps(self, start : str, end : str, offset=0, tSubgoal_flag=True):
        state_map = [np.array([], dtype=int) for _ in range(self.num_trials)]
        for trial in range(self.num_trials):
            diff = self.beh_data[end].iloc[trial] - self.beh_data[start].iloc[trial]
            # Check if tSubgoalResp is not an empty array
            t_subgoal = self.beh_data['tSubgoalOn'].iloc[trial]
            if tSubgoal_flag and isinstance(t_subgoal, np.ndarray) and t_subgoal.size > 0:
                diff = self.beh_data['tSubgoalOn'].iloc[trial] - self.beh_data['tFix'].iloc[trial]
            x = np.arange(0, diff)
            t_tmp = self.beh_data['tFix'].iloc[trial] - self.t_beginning
            state_map[trial] = x + t_tmp + offset
            state_map[trial] = state_map[trial].astype(int)

        return state_map
       
