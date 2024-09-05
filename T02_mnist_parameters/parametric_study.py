from mnist_nn_custom import mnist_nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

param_names = [
    'number_of_hidden_layers',
    'activation_functions',
    'loss functions',
    #'regularizations',
    'batch_sizes',
    'number_of_epochs',
    'learning_rates'
]

param_labels = [
    ['3 hidden_layers', '5 hidden_layers'],
    ['sigmoid', 'tanh'],
    ['mse_loss', 'cross_entropy'],
    #['L1', 'L2'],
    ['16 batch_size', '256 batch_size'],
    ['25 epochs'],
    ['lr = 0.1', 'lr = 10.0']
]

ref_labels = [
    '1 hidden_layer',
    'relu',
    'nll_loss',
    #'None',
    '64 batch_size',
    '14 epochs',
    'lr = 1.0'
]

case_params = [
    # test with different number of hidden layers
    [
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128, 64, 64], [F.relu, F.relu, F.relu], F.nll_loss, [None, None, None]],
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [256, 128, 128, 64, 64], [F.relu, F.relu, F.relu, F.relu, F.relu], F.nll_loss, [None, None, None, None, None]],
    ],
    # test with different activation functions
    [
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.sigmoid], F.nll_loss, [None]],
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.tanh], F.nll_loss, [None]],
    ],
    # test with different loss functions
    [
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.mse_loss, [None]],
        [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.cross_entropy, [None]],
    ],
    ## test with different regularizations
    #[
    #    [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [F.l1_loss]],
    #    [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [0.01]],
    #],
    # test with different batch sizes
    [
        [16, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]],
        [256, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]],
    ],
    # test with different number of epochs
    [
        [64, 1000, 25, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]],
    ],
    # test with different learning rates
    [
        [64, 1000, 14, 0.1, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]],
        [64, 1000, 14, 10.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]],
    ]
]

fruntime = open('runtime.txt', 'w', buffering=1)

ref_params = [64, 1000, 14, 1.0, False, False, False, 1, 40, False, [128], [F.relu], F.nll_loss, [None]]
metrics_ref = mnist_nn(*ref_params)

fruntime.write(f'Case 0: reference run time = {metrics_ref.run_time}\n')

# Call the main function with desired parameters
for idx, params in enumerate(case_params):
    metrics = []
    for i, param in enumerate(params):
        imet = mnist_nn(*param)
        metrics.append(imet)
        fruntime.write(f'Case {idx+1}: {param_labels[idx][i]} run time = {imet.run_time}\n')

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Parameter Study: {param_names[idx]}')
    axs[0].plot(metrics_ref.train_count, metrics_ref.train_loss, label=ref_labels[idx])
    for i, metric in enumerate(metrics):
        axs[0].plot(metric.train_count, metric.train_loss, label=param_labels[idx][i])
    axs[0].legend()
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('data count')
    axs[0].set_ylabel('loss')
    axs[0].grid()

    axs[1].plot(metrics_ref.test_count, metrics_ref.test_accuracy, label=ref_labels[idx])
    for i, metric in enumerate(metrics):
        axs[1].plot(metric.test_count, metric.test_accuracy, label=param_labels[idx][i])
    axs[1].legend()
    axs[1].set_title('Test Accuracy')
    axs[1].set_xlabel('data count')
    axs[1].set_ylabel('accuracy')
    axs[1].grid()

    plt.savefig(f'test{idx+1}_{param_names[idx]}.png')

fruntime.close()
