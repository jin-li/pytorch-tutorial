import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from SineWaveDataset import create_dataset, RNNDataset

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer is output of model

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Return only the last output of RNN.

        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer is output of model
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the GRU
        return out

class PredictResult:
    def __init__(self, initial_x, initial_y, predict_x, predict_y, exact_x, exact_y):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.predict_x = predict_x
        self.predict_y = predict_y
        self.exact_x = exact_x
        self.exact_y = exact_y

class Result:
    def __init__(self, train_loss, test_loss, predict_result):
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.predict_result = predict_result

def train(model, device, train_dataloader, loss_function, optimizer, epoch_idx, log_interval):
    model.train()
    trained_cnt = 0
    loss_all = []
    for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()

        trained_cnt += len(x_batch)
        if batch_idx % log_interval == 0 or batch_idx == len(train_dataloader) - 1:
            print('Train Epoch: {:5d} [{:5d} / {:5d} ({:3.0f}%)]\tLoss: {:.3e}'.format(
                    epoch_idx, trained_cnt, len(train_dataloader.dataset),
                    100. * (batch_idx + 1) / len(train_dataloader), loss.item())
            )
    return np.mean(loss_all)

def test(model, device, test_dataloader, loss_function):
    model.eval()
    loss_all = []

    for x_batch, y_batch in test_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss_all.append(loss.cpu().data.numpy())

    print('Test loss: ', np.mean(loss_all))
    return np.mean(loss_all)

def predict(model, device, predict_start, prediction_steps, look_back=80):
    look_back = 80
    initial = predict_start - look_back/2
    X_test = np.arange(initial, predict_start, 0.5)
    y_test = np.sin(X_test * 2 * np.pi / 20)

    predict_series = y_test[:80]
    predict_series = torch.from_numpy(predict_series).float().to(device)  # Move data to GPU
    predict_dataset = torch.stack([predict_series]).unsqueeze(-1).to(device)  # Move data to GPU

    predictions = []
    for i in range(prediction_steps):
        with torch.no_grad():
            prediction = model(predict_dataset).squeeze().item()
            predictions.append(prediction)
            predict_series = torch.cat((predict_series[1:], torch.tensor([prediction]).to(device)))  # Move data to GPU
            predict_dataset = torch.stack([predict_series]).unsqueeze(-1).to(device)  # Move data to GPU
    
    x = np.arange(predict_start, predict_start+prediction_steps/2, 0.5)
    y_exact = np.sin(x * 2 * np.pi / 20)

    return PredictResult(X_test, y_test, x, predictions, x, y_exact)

def time_series_models(device='cuda', model_type='rnn', hidden_size=20, batch_size=500, test_batch_size=200, epochs=500, lr=0.001, predict_start=123, prediction_steps=100, plot=False):

    if model_type == 'RNN':
        model = SimpleRNN(hidden_size=hidden_size).float().to(device)
    elif model_type == 'LSTM':
        model = SimpleLSTM(hidden_size=hidden_size).float().to(device)
    elif model_type == 'GRU':
        model = SimpleGRU(hidden_size=hidden_size).float().to(device)
    else:
        exit('Invalid model type. Please choose one of the following: RNN, LSTM, GRU')
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists('train_data.pt') and os.path.exists('test_data.pt'):
        print('Loading train data \"train_data.pt\" and test data \"test_data.pt\"')
        train_data = torch.load('train_data.pt', weights_only=False)
        test_data = torch.load('test_data.pt', weights_only=False)
    else:
        exit('No train data or test data found. Please generate data first by running the script with the --generate-data flag.')

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    train_loss = []
    test_loss = []
    for epoch in range(1, epochs+1):
        train_loss_tmp = train(model, device, dataloader, loss_function, optimizer, epoch, 10)
        train_loss.append(train_loss_tmp)
        test_loss_tmp = test(model, device, test_dataloader, loss_function)
        test_loss.append(test_loss_tmp)

    predict_result = predict(model, device, predict_start, prediction_steps)
    
    if plot:
        # Plot the training loss
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(train_loss, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss (MSE)")
        ax.set_yscale('log')
        plt.savefig(f'training_loss_{model_type}.png')
        plt.close()

        # Plot the test loss
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(test_loss, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Loss (MSE)")
        ax.set_yscale('log')
        plt.savefig(f'test_loss_{model_type}.png')
        plt.close()

        # Plot the prediction
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(predict_result.initial_x, predict_result.initial_y, lw=3, c='b', label='initial data')
        ax.plot(predict_result.exact_x, predict_result.exact_y, lw=3, c='g', label='exact data')
        ax.scatter(predict_result.predict_x, predict_result.predict_y, c='r', marker='x', label='predictions')
        ax.legend(loc="upper right")
        plt.savefig(f'sine_prediction_{model_type}.png')
        plt.close()
    
    return Result(train_loss, test_loss, predict_result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time Series Example')
    parser.add_argument('--model-type', type=str, default='rnn', metavar='MODEL',
                        help='model type (available: RNN, LSTM, GRU; default: RNN)')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                        help='hidden size of the RNN (default: 20)')
    parser.add_argument('--generate-data', action='store_true', default=False,
                        help='generate new data')
    parser.add_argument('--sequence-length', type=int, default=80, metavar='N',
                        help='sequence length for the time series (default: 80)')
    parser.add_argument('--predict-steps', type=int, default=100, metavar='N',
                        help='number of prediction steps (default: 100)')
    parser.add_argument('--predict-start', type=int, default=123, metavar='N',
                        help='start of prediction (default: 123)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot the prediction')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif use_mps:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    if args.generate_data:
        create_dataset(sequence_length=args.sequence_length, train_percent=0.8)
    
    time_series_models(device=device, model_type=args.model_type, hidden_size=args.hidden_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, predict_start=args.predict_start, prediction_steps=args.predict_steps, plot=args.plot)