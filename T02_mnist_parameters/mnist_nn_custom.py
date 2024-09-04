import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

class CustomNN(nn.Module):
    def __init__(self, hidden_layers, activation_functions, regularizations):
        super(CustomNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.activation_functions = activation_functions
        self.regularizations = regularizations
        input_size = 28 * 28  # Input layer size

        # Create hidden layers
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(input_size, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.activation_functions[i]:
                x = self.activation_functions[i](x)  # Apply activation function
            if self.regularizations[i]:
                x = self.regularizations[i](x)  # Apply regularization
        x = self.output_layer(x)  # Output layer
        return F.log_softmax(x, dim=1)  # Apply log-softmax for classification

class PerformanceMetrics:
    def __init__(self):
        self.train_count = []
        self.train_loss = []
        self.test_count = []
        self.test_loss = []
        self.test_accuracy = []
        self.run_time = 0

def train(args, model, device, train_loader, optimizer, epoch, loss_function, metrics):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # If using mse_loss, convert target to one-hot encoding
        if loss_function == F.mse_loss:
            target = F.one_hot(target, num_classes=10).float()
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args['log_interval'] == 0:
            metrics.train_loss.append(loss.item())
            if metrics.train_count == []:
                metrics.train_count.append(0)
            else:
                metrics.train_count.append(metrics.train_count[-1]+args['batch_size']*args['log_interval'])

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break

def test(model, device, test_loader, loss_function, metrics):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # If using mse_loss, convert target to one-hot encoding
            if loss_function == F.mse_loss:
                target = F.one_hot(target, num_classes=10).float()
            
            output = model(data)
            # Handle reduction for mse_loss
            if loss_function == F.mse_loss:
                test_loss += loss_function(output, target).sum().item()
            else:
                test_loss += loss_function(output, target, reduction='sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            if loss_function == F.mse_loss:
                correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
            else:
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if metrics.test_count == []:
        metrics.test_count.append(len(test_loader.dataset))
    else:
        metrics.test_count.append(metrics.test_count[-1] + len(test_loader.dataset))
    metrics.test_loss.append(test_loss)
    metrics.test_accuracy.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

def mnist_nn(batch_size           = 64, 
         test_batch_size      = 1000, 
         epochs               = 14, 
         lr                   = 1.0, 
         no_cuda              = False, 
         no_mps               = False, 
         dry_run              = False, 
         seed                 = 1, 
         log_interval         = 10, 
         save_model           = False, 
         hidden_layers        = [128], 
         activation_functions = [F.relu], 
         loss_function        = F.nll_loss, 
         regularizations      = [None]
         ):
    start_time = time.time()
    metrics = PerformanceMetrics()

    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CustomNN(hidden_layers, activation_functions, regularizations).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    # Check if any element in regularizations is a float
    weight_decay = 0.0
    for reg in regularizations:
        if isinstance(reg, float):
            weight_decay = reg
            break

    # Create the model and optimizer
    model = CustomNN(hidden_layers, activation_functions, [None if isinstance(reg, float) else reg for reg in regularizations]).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

    args = {
        'dry_run': dry_run,
        'batch_size': batch_size,
        'log_interval': log_interval
    }

    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, loss_function, metrics)
        test(model, device, test_loader, loss_function, metrics)

    if save_model:
        torch.save(model.state_dict(), "mnist_custom_nn.pt")
    
    end_time = time.time()
    metrics.run_time = end_time - start_time
    return metrics

# Example of how to call the main function from another script
if __name__ == '__main__':
    mnist_nn()