"""
Null hypothesis testing for neural decoders using support library.

Tests whether neural activity significantly decodes behavioral variables
by comparing real decoding accuracy against a null distribution created
by shuffling labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from support.beh import Behavior
from support.ephys import Spikes
from support.linear_decoder import run_linear_decoder


def load_and_process_subjects(group: str, region: str, decoder_type: str, subjs: list = None, state_map_fixed=True):
    """
    Load and process behavioral and ephys data for all subjects.
    
    Args:
        group: Data directory name (e.g., 'behEphys')
        region: Brain region to analyze (e.g., 'ACC', 'H', 'OF')
        decoder_type: Type of decoder ('second_choice' or 'option')
        subjs: Optional list of subject IDs to process
    
    Returns:
        tuple: (spikes_list, labels_list, min_labels, subjects_with_cells)
    """
    # Get subject folders
    if subjs is None:
        current_dir = Path.cwd()
        data_dir = current_dir / group
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        subjs = [d.name for d in data_dir.iterdir() if d.is_dir() and 'subj' in d.name]
    
    n_subjs = len(subjs)
    
    # Initialize storage
    spikes = [None] * n_subjs
    labels = [None] * n_subjs
    min_labels = np.zeros(n_subjs)
    
    # Process each subject
    for i, subj in enumerate(subjs):
        try:
            subj_beh = Behavior(subj)
            subj_ephys = Spikes(subj, subj_beh.t_beginning, 
                               subj_beh.t_session, subj_beh.num_trials)
            
            # Create labels based on decoder type
            if decoder_type == 'second_choice':
                labels[i] = subj_beh.get_feature_values('choice2')
                n_labels = 2
            elif decoder_type == 'option':
                choice1 = subj_beh.get_feature_values('choice1')
                choice2 = subj_beh.get_feature_values('choice2')
                # Combine choices into 4 options: 1=LL, 2=LR, 3=RL, 4=RR
                labels[i] = (choice1 - 1) * 2 + choice2
                n_labels = 4
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")
            
            # Count cells in region
            num_cells_region = sum(1 for cell_name in subj_ephys.cell_names
                                  if subj_ephys.get_region(cell_name, full_region=False) == region)
            
            if num_cells_region > 0:
                # Calculate minimum trials per label
                if decoder_type == 'option':
                    min_labels[i] = np.min(np.histogram(labels[i], bins=n_labels)[0])
                else:
                    min_labels[i] = min(np.sum(labels[i] == 1), np.sum(labels[i] == 2))
                
                #state_map = subj_beh.construct_timestep_index_maps('tFix', 1001, offset=0)
                if state_map_fixed:
                    state_map = subj_beh.construct_timestep_index_maps('tFix', 1000, 200)
                else:
                    state_map = subj_beh.construct_nonfixed_timestep_index_maps('tFix', 'tState1On', 200)
                
                # Extract and z-score spike counts for each cell
                for cell_name in subj_ephys.cell_names:
                    if subj_ephys.get_region(cell_name, full_region=False) == region:
                        cell_spikes = subj_ephys.z_score_cell(state_map, cell_name, fixed=state_map_fixed) #Change Fixed depending on STATE_MAP
                        
                        if cell_spikes is not None:
                            # Accumulate cells for this subject
                            cell_spikes = cell_spikes.reshape(1, -1)
                            if spikes[i] is None:
                                spikes[i] = cell_spikes
                            else:
                                spikes[i] = np.vstack((spikes[i], cell_spikes))
        
        except Exception as e:
            print(f"Warning: Error processing {subj}: {e}")
            continue
    
    # Filter subjects with cells in region
    subjects_with_cells = np.where(min_labels > 0)[0]
    
    if len(subjects_with_cells) == 0:
        raise ValueError(f"No subjects have cells in region {region}")
    
    
    return spikes, labels, min_labels, subjects_with_cells


def test_decoder(decoder_type: str, region: str = 'ACC', group: str = 'behEphys',
                subjs: list = None, n_perms: int = 1000, n_folds: int = 5):
    """
    Run permutation test comparing real vs. null decoding accuracy.
    
    Tests whether neural activity significantly decodes behavioral variables
    by comparing real decoding accuracy against a null distribution created
    by shuffling labels.
    
    Args:
        decoder_type: Type of decoder ('second_choice' or 'option')
        region: Brain region to analyze (default: 'ACC')
        group: Data directory name (default: 'behEphys')
        subjs: Optional list of subject IDs to process
        n_perms: Number of permutations for null distribution (default: 1000)
        n_folds: Number of CV folds (default: 5)
    
    Returns:
        dict: Results dictionary with keys:
            - real_acc: Real decoding accuracy
            - null_accuracies: Array of null accuracies (length n_perms)
            - null_mean: Mean of null distribution
            - null_std: Standard deviation of null distribution
            - p_value: Exact empirical p-value
            - significant: Boolean indicating p < 0.05
    """
    print(f"Running permutation test for {decoder_type} in {region}")
    print(f"Building null distribution with {n_perms} permutations...")
    
    # Load data once
    spikes, labels, min_labels, subjects_with_cells = load_and_process_subjects(
        group, region, decoder_type, subjs
    )
    
    n_labels = 2 if decoder_type == 'second_choice' else 4
    n_subsamp = int(np.min(min_labels[subjects_with_cells]))
    
    # Build null distribution (shuffled labels)
    print("Running null decoder (shuffled labels)...")
    null_accuracies, _ = run_linear_decoder(
        n_perms=n_perms,
        n_folds=n_folds,
        subjs_with_cells=subjects_with_cells,
        spikes=spikes,
        n_subsamp=n_subsamp,
        n_labels=n_labels,
        choice_labels=labels,
        shuffle_labels=True
    )
    
    # Run real decoding
    print("Running real decoder...")
    real_accuracies, _ = run_linear_decoder(
        n_perms=n_perms,
        n_folds=n_folds,
        subjs_with_cells=subjects_with_cells,
        spikes=spikes,
        n_subsamp=n_subsamp,
        n_labels=n_labels,
        choice_labels=labels,
        shuffle_labels=False
    )
    
    # Compute statistics
    real_acc = np.mean(real_accuracies)
    p_value = np.mean(null_accuracies >= real_acc)
    null_mean = np.mean(null_accuracies)
    null_std = np.std(null_accuracies)
    significance = p_value < 0.05
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Permutation Test Results: {decoder_type} in {region}")
    print('='*60)
    print(f"Real accuracy:        {real_acc:.4f}")
    print(f"Null mean ± std:      {null_mean:.4f} ± {null_std:.4f}")
    print(f"p-value:              {p_value:.4f} ({np.sum(null_accuracies >= real_acc)}/{n_perms})")
    print(f"Significant (p<0.05): {significance}")
    print('='*60)
    
    return {
        'real_acc': real_acc,
        'null_accuracies': null_accuracies,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_value': p_value,
        'significant': significance
    }


def compare_decoders(decoder_type: str, regions: list = None, group: str = 'behEphys',
                    n_perms: int = 1000, n_folds: int = 5):
    """
    Compare real vs. null decoding across multiple brain regions with exact p-values.
    
    Args:
        decoder_type: Type of decoder ('second_choice' or 'option')
        regions: List of brain regions to test (default: all 9 regions)
        group: Data directory name (default: 'behEphys')
        n_perms: Number of permutations for null distribution (default: 1000)
        n_folds: Number of CV folds (default: 5)
    
    Returns:
        pd.DataFrame: Results table with columns:
            - Region: Brain region name
            - Real Acc: Real decoding accuracy
            - Null Mean: Mean of null distribution
            - Null Std: Standard deviation of null distribution
            - p-value: Exact empirical p-value (proportion null >= real)
            - Significant: Boolean indicating p < 0.05
    """
    if regions is None:
        regions = ["ACC", "A", "H", "OF", "PT", "CM", "SMA", "PRV", "IFG"]
    
    results = []
    
    for region in regions:
        print(f"\n{'='*60}")
        print(f"Testing {decoder_type} decoder in {region}")
        print('='*60)
        
        try:
            test_results = test_decoder(
                decoder_type=decoder_type,
                region=region,
                group=group,
                n_perms=n_perms,
                n_folds=n_folds
            )
            
            results.append({
                'Region': region,
                'Real Acc': round(test_results['real_acc'], 4),
                'Null Mean': round(test_results['null_mean'], 4),
                'Null Std': round(test_results['null_std'], 4),
                'p-value': round(test_results['p_value'], 4),
                'Significant': test_results['significant']
            })
        
        except Exception as e:
            print(f"Error processing {region}: {e}")
            results.append({
                'Region': region,
                'Real Acc': np.nan,
                'Null Mean': np.nan,
                'Null Std': np.nan,
                'p-value': np.nan,
                'Significant': False
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"{decoder_type.upper()} Decoding Results (Real vs. Null)")
    print('='*60)
    print(results_df.to_string(index=False))
    
    return results_df


