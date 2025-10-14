import numpy as np


from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix



def build_perm_spikes(n_subsamp : int, n_labels : int, choice_labels : list[int], spikes : list[np.ndarray], perm_spikes : list[np.ndarray], shuffle_labels=False):
    if shuffle_labels:
        choice_labels = np.random.permutation(choice_labels)

    for cI in range(spikes.shape[0]):
        perm_choices = np.empty(n_subsamp*n_labels)
        for c_idx in range(n_labels):
            c_tmp = np.where(choice_labels == c_idx+1)[0] #index of the trial that has second choice c_idx+1 (L or R))
            c_tmp = np.random.permutation(c_tmp)[:n_subsamp] #takes first n_subsamp trials
            perm_choices[(c_idx)*n_subsamp:(c_idx+1)*n_subsamp] = c_tmp #stores indices of random trials of the second choice (L or R) for each patient

        perm_choices = perm_choices.astype(int)
        if perm_spikes is None:
            perm_spikes = spikes[cI, perm_choices].reshape(-1, 1)
        else:
            included_spikes = spikes[cI, perm_choices].reshape(-1, 1)  # transpose to match hstack format
            perm_spikes = np.hstack((perm_spikes, included_spikes))
        
    return perm_spikes

def run_linear_decoder(n_perms: int, n_folds: int, subjs_with_cells: np.ndarray, 
                       spikes: list[np.ndarray], n_subsamp: int, n_labels: int, 
                       choice_labels: list[int], shuffle_labels: bool = False):
    """
    Run linear decoder with cross-validation and permutations.
    
    Args:
        n_perms: Number of permutations
        n_folds: Number of CV folds
        subjs_with_cells: Indices of subjects with cells
        spikes: List of spike arrays per subject
        n_subsamp: Number of samples per label
        n_labels: Number of labels to decode
        choice_labels: Labels for each subject
        shuffle_labels: If True, shuffle labels for null hypothesis
    
    Returns:
        tuple: (accuracies, within_class_acc)
            - accuracies: Array of shape (n_perms,) with accuracy per permutation
            - within_class_acc: Array of shape (n_perms, n_labels) with per-class accuracies
    """
    choices = np.concatenate([np.ones(n_subsamp) * (i+1) for i in range(n_labels)])

    accuracies = np.zeros(n_perms)
    within_class_acc = np.zeros((n_perms, n_labels))
    
    for p_idx in range(n_perms):
        perm_spikes = None
        for subj_i in subjs_with_cells: 
            if spikes[subj_i].size > 0:
                if spikes[subj_i].ndim == 1:
                    spikes[subj_i] = spikes[subj_i].reshape(1, -1)
                
                perm_spikes = build_perm_spikes(n_subsamp, n_labels, choice_labels[subj_i], 
                                               spikes[subj_i], perm_spikes, shuffle_labels)
                
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        perm_accuracies = np.zeros(n_folds)
        perm_within_class_acc = np.zeros((n_folds, n_labels))

        for f_idx, (train_idx, test_idx) in enumerate(skf.split(perm_spikes, choices)):
            X_train, X_test = perm_spikes[train_idx, :], perm_spikes[test_idx, :]
            y_train, y_test = choices[train_idx], choices[test_idx]

            model = LinearSVC(penalty="l2", C=1, max_iter=10000)
            model = model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            perm_accuracies[f_idx] = accuracy_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            perm_within_class_acc[f_idx, :] = cm.diagonal() / cm.sum(axis=1)

        accuracies[p_idx] = np.mean(perm_accuracies)
        within_class_acc[p_idx] = np.mean(perm_within_class_acc, axis=0)

    return accuracies, within_class_acc
            


        
        
