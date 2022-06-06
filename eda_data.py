from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle as pickle


# set the randomness to keep reproducible results
# torch.manual_seed(0)
# np.random.seed(0)

# READING DATA #

ROOT_PATH = './'


def get_city_trajectories(city="palo-alto"):
    assert city in ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]

    train_inpath = f'{ROOT_PATH}train/{city}_inputs'
    train_outpath = f'{ROOT_PATH}train/{city}_outputs'
    test_inpath = f'{ROOT_PATH}train/{city}_inputs'

    train_inputs = np.asarray(pickle.load(open(train_inpath, "rb")))
    train_outputs = np.asarray(pickle.load(open(train_outpath, "rb")))
    test_inputs = np.asarray(pickle.load(open(test_inpath, "rb")))

    return train_inputs, train_outputs, test_inputs


def train_val_split(inputs, outputs, train_pct=.8, rand_split=False):
    assert isinstance(train_pct, (int, float)) and 0 < train_pct <= 1
    assert rand_split in [True, False]

    if not rand_split:
        train_size = int(len(inputs) * train_pct)
        return (
            (inputs[:train_size], outputs[:train_size]),
            (inputs[train_size:], outputs[train_size:])
        )
    input_train, input_test, output_train, output_test = train_test_split(
        inputs, outputs, test_size=(1 - train_pct)
    )
    return (input_train, output_train), (input_test, output_test)


class ArgoverseDataset(Dataset):

    def __init__(self, inputs, outputs=None, split='train', city=None):
        super(ArgoverseDataset, self).__init__()
        self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs
        self.split = split
        self.normalized = False
        self.start_pos = self.inputs[:, 0, :].copy()
        self.city = city

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.split in ['train', 'val']:
            return self.inputs[idx], self.outputs[idx]
        elif self.split == 'test':
            return self.inputs[idx]

    def normalize(self, keep_train_start=False):
        if self.normalized:
            return

        # normalize inputs
        for i in range(len(self)):
            self.inputs[i] -= self.start_pos[i, :]

        # normalize outputs
        if self.split in ["train", "val"]:
            for i in range(len(self)):
                self.outputs[i] -= self.start_pos[i, :]

    def transform(self):
        new_inputs = []
        new_outputs = []
        for agent_in, agent_out in tqdm(zip(self.inputs, self.outputs)):
            arr = np.concatenate((agent_in, agent_out), axis=0)
            df = pd.DataFrame(arr, columns=['x', 'y'])

            # add velocity vector and speed
            df['x_vel'] = np.gradient(df.x)
            df['y_vel'] = np.gradient(df.y)
            df['vel'] = np.sqrt(df.x_vel ** 2 + df.y_vel ** 2)

            # acceleration
            df['x_acc'] = np.gradient(df.x_vel)
            df['y_acc'] = np.gradient(df.y_vel)
            df['acc'] = np.gradient(df.vel)
            # tangent = np.array([1 / df.vel] * 2).T * np.array([df.x_vel, df.y_vel]).T

            # curvature
            df['curvature'] = np.abs(df.x_acc * df.y_vel - df.x_vel * df.y_acc) / df.vel ** 3

            out = df[['x', 'y', 'curvature']].to_numpy()

            new_inputs.append(out[:50, :])
            new_outputs.append(out[50:, :])
        self.inputs = np.array(new_inputs)
        self.outputs = np.array(new_outputs)



