import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import pickle as pickle
from glob import glob
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


# PLOTTING TRAJECTORIES #

def show_sample_batch(sample_batch):
    """visualize the trajectory for a batch of samples"""
    inp, out = sample_batch
    batch_sz = inp.size(0)
    agent_sz = inp.size(1)

    fig, axs = plt.subplots(1, batch_sz, figsize=(15, 3), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    for i in range(batch_sz):
        axs[i].xaxis.set_ticks([])
        axs[i].yaxis.set_ticks([])

        # first two feature dimensions are (x,y) positions
        axs[i].scatter(inp[i, :, 0], inp[i, :, 1])
        axs[i].scatter(out[i, :, 0], out[i, :, 1])
    axs[0].legend(handles=[
        Line2D(
            [0], [0], marker='o', color='w', label='Input',
            markerfacecolor='C0', markersize=10
        ),
        Line2D(
            [0], [0], marker='o', color='w', label='Output',
            markerfacecolor='C1', markersize=10
        )
    ])


def show_point_distributions(dataset, bins=None):
    if bins is None:
        bins = [120, 100]

    all_inputs = np.reshape(dataset.inputs, (-1, 2))
    if dataset.city is None:
        in_header = 'Input Distribution'
        out_header = 'Output Distribution'
    else:
        in_header = r"$\bf{" + dataset.city + "}$" + ' Input Distribution'
        out_header = r"$\bf{" + dataset.city + "}$" + ' Output Distribution'

    if dataset.split in ['train', 'val']:
        # get outputs
        all_outputs = np.reshape(dataset.outputs, (-1, 2))

        # setup plots
        fig, ax = plt.subplots(1, 2, figsize=(16, 4), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.05)
        ax = ax.ravel()

        # plot inputs
        in_map = ax[0].hist2d(all_inputs[:, 0], all_inputs[:, 1], bins=bins, density=True)
        fig.colorbar(in_map[-1], ax=ax[0])
        ax[0].set_title(in_header, fontsize=16)

        # plot outputs
        out_map = ax[1].hist2d(all_outputs[:, 0], all_outputs[:, 1], bins=bins, density=True)
        fig.colorbar(out_map[-1], ax=ax[1])
        ax[1].set_title(out_header, fontsize=16)

    elif dataset.split == 'test':
        plt.hist2d(all_inputs[:, 0], all_inputs[:, 1], bins=bins, density=True)
        plt.colorbar()
        plt.title('Input Distribution')


def show_starting_positions(dataset, bins=None):
    if bins is None:
        bins = [120, 100]

    start_positions = np.reshape(dataset.start_pos, (-1, 2))
    if dataset.city is None:
        in_header = 'Input Distribution'
    else:
        in_header = r"$\bf{" + dataset.city + "}$" + ' Starting Positions'

    plt.hist2d(start_positions[:, 0], start_positions[:, 1], bins=bins, density=True, cmap='plasma')
    plt.colorbar()
    plt.title(in_header, fontsize=16)


