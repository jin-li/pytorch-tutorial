import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

class PredictResult:
    def __init__(self, predict_start, prediction_steps, predictions):
        self.predict_start = predict_start
        self.prediction_steps = prediction_steps
        self.predictions = predictions

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
        if batch_idx % log_interval == 0:
            print('Train Epoch: {:5d} [{:5d} / {:5d} ({:3.0f}%)]\tLoss: {:.3e}'.format(
                    epoch_idx, trained_cnt, len(train_dataloader.dataset),
                    100. * (batch_idx + 1) / len(train_dataloader), loss.item())
            )
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

    return PredictResult(predict_start, prediction_steps, predictions)

def time_series_rnn(device, hidden_size=20, batch_size=500, epochs=500, lr=0.001, predict_start=123, prediction_steps=100, plot=True):
    model = SimpleRNN(hidden_size=hidden_size).float().to(device)  # Move model to GPU
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = torch.load('train_data.pt', weights_only=False)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    loss_curve = []
    for epoch in range(1, epochs+1):
        loss_tmp = train(model, device, dataloader, loss_function, optimizer, epoch, 10)
        loss_curve.append(loss_tmp)
    
    if plot:
        # Plot the training loss
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(loss_curve, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss (MSE)")
        ax.set_yscale('log')
        plt.savefig('train_loss_rnn.png')
        plt.close()

        # Plot the prediction
        look_back = 80
        predict_result = predict(model, device, predict_start, prediction_steps)
        initial = predict_start - look_back/2
        X_test = np.arange(initial, predict_start, 0.5)
        y_test = np.sin(X_test * 2 * np.pi / 20)
        x = np.arange(predict_start, predict_start+prediction_steps/2, 0.5)
        y_exact = np.sin(x * 2 * np.pi / 20)
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(X_test, y_test, lw=3, c='b', label='initial data')
        ax.plot(x, y_exact, lw=3, c='g', label='exact data')
        ax.scatter(x, predict_result.predictions, c='r', marker='x', label='predictions')
        ax.legend(loc="upper right")
        plt.savefig('sine_plot_rnn.png')
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time Series Example')
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif use_mps:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    time_series_rnn(device=device, hidden_size=args.hidden_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, predict_start=args.predict_start, prediction_steps=args.predict_steps)