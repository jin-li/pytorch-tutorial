import numpy as np
import torch
from torch.utils.data import Dataset

class RNNDataset(Dataset):

    def __init__(self, x, y=None):
        self.data = x
        self.labels = y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

def create_dataset(sequence_length, train_percent=0.8):

    # Create sin wave at discrete time steps.
    num_time_steps = 2000
    time_steps = np.linspace(start=0, stop=1000, num=num_time_steps, dtype=np.float32)
    discrete_sin_wave = (np.sin(time_steps * 2 * np.pi / 20)).reshape(-1, 1)

    # Take (sequence_length + 1) elements & put as a row in sequence_data, extra element is value we want to predict.
    # Move one time step and keep grabbing till we reach the end of our sampled sin wave.
    sequence_data = []
    for i in range(num_time_steps - sequence_length):
        sequence_data.append(discrete_sin_wave[i: i + sequence_length + 1, 0])
    sequence_data = np.array(sequence_data)

    # Split for train/val.
    num_total_samples = sequence_data.shape[0]
    num_train_samples = int(train_percent * num_total_samples)

    train_set = sequence_data[:num_train_samples, :]
    test_set = sequence_data[num_train_samples:, :]

    print('{} total sequence samples, {} used for training'.format(num_total_samples, num_train_samples))

    # Take off the last element of each row and this will be our target value to predict.
    x_train = train_set[:, :-1][:, :, np.newaxis]
    y_train = train_set[:, -1][:, np.newaxis]
    x_test = test_set[:, :-1][:, :, np.newaxis]
    y_test = test_set[:, -1][:, np.newaxis]

    train_data = RNNDataset(x_train, y_train)
    test_data = RNNDataset(x_test, y_test)

    torch.save(train_data, 'train_data.pt')
    torch.save(test_data, 'test_data.pt')

if __name__ == '__main__':
    create_dataset(sequence_length=80)