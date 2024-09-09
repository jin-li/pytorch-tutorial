from mnist_cnn import mnist_cnn
import matplotlib.pyplot as plt

metrics = mnist_cnn()
print(f'MNIST CNN runtime: {metrics.run_time}')

# Plot the training loss and test accuracy
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].plot(metrics.train_count, metrics.train_loss)
axs[0].set_title('Training Loss')
axs[0].set_xlabel('data count')
axs[0].set_ylabel('loss')
axs[0].grid()

axs[1].plot(metrics.test_count, metrics.test_accuracy)
axs[1].set_title('Test Accuracy')
axs[1].set_xlabel('data count')
axs[1].set_ylabel('accuracy')
axs[1].grid()

plt.savefig(f'mnist_cnn_performance.png')