"""
created 25.4.9

plot the decoding results

"""

import numpy as np
import matplotlib.pyplot as plt
#from decode_subgoal_mcsd import decode_subgoal_mcsd
from decode_second_choice_mcsd import decode_second_choice_mcsd
from decode_option_mcsd import decode_option_mcsd

def plot_decoding_mcsd(decode_type='option',group='behEphys',regions=['SMA','ACC','OF','A','H','PT','CM','PRV', 'IFG'],subjs=[]):
    n_regions = len(regions)
    accuracies = [None]*n_regions
    within_class_accs = [None]*n_regions

    for r_idx, region in enumerate(regions):
        print(f"Processing region: {region}")
        #if decode_type == 'subgoal':
        #    region_accuracies, region_within_class_accs = decode_subgoal_mcsd(group=group,region=region,subjs=subjs)
        if decode_type == 'second_choice':
            region_accuracies, region_within_class_accs = decode_second_choice_mcsd(group=group,region=region,subjs=subjs)
        elif decode_type == 'option':
            region_accuracies, region_within_class_accs = decode_option_mcsd(group=group,region=region,subjs=subjs)

        accuracies[r_idx] = region_accuracies
        within_class_accs[r_idx] = region_within_class_accs
        print(f"Completed region: {region}")

    if decode_type == 'subgoal':
        chance_acc = 1/3
        y_lims = (0.1, 0.7)
        n_epochs = len(accuracies[0])
    elif decode_type == 'second_choice':
        chance_acc = 1/2
        y_lims = (0.3, 0.8)
        n_epochs = 1
    elif decode_type == 'option':
        chance_acc = 1/4
        y_lims = (0.1, 0.7)
        n_epochs = 1

    epoch_labels = [f'state {i+1}' for i in range(n_epochs)]
    epoch_qtls = [(0.25, 0.75) for i in range(n_epochs)]

    fig, ax = plt.subplots(1, n_regions, figsize=(n_regions*4, 4))
    colors = plt.cm.cool(np.linspace(0, 1, n_regions))
    for r_idx in range(n_regions):
        plt.subplot(1, n_regions, r_idx+1)
        region_acc = plt.violinplot(accuracies[r_idx], showmeans=True, showmedians=False, quantiles=epoch_qtls, showextrema=False)
        for vp in region_acc['bodies']:
            vp.set_facecolor(colors[r_idx])
        region_acc['cmeans'].set_color('black')
        x_lims = ax[r_idx].get_xlim()
        plt.plot(x_lims, [chance_acc, chance_acc], color='k', linestyle='--')
        ax[r_idx].set_xlim(x_lims)
        ax[r_idx].set_ylim(y_lims)

        plt.xticks(np.linspace(1,n_epochs,num=n_epochs), epoch_labels)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        ax[r_idx].spines['top'].set_visible(False)
        ax[r_idx].spines['right'].set_visible(False)
        plt.title(regions[r_idx])
    plt.show()

    print("Starting to create plots...")

if __name__ == "__main__":
    plot_decoding_mcsd()
    #plot_decoding_mcsd(decode_type='second_choice')
