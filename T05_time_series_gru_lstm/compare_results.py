import matplotlib.pyplot as plt
from time_series_models import time_series_models, Result

models = ['RNN', 'LSTM', 'GRU']
markers = ['o', '+', 'x']

results = []
for model in models:
    result = time_series_models(model_type=model)
    results.append(result)

# Plot the train and test loss
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.title.set_text('Train/Test Loss')
for i, result in enumerate(results):
    ax.plot(result.train_loss, label=models[i]+' train')
    ax.plot(result.test_loss, label=models[i]+' test', linestyle='--')
ax.legend(loc='upper right')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid()
ax.set_yscale('log')
plt.savefig('train_test_loss.png')
plt.close()

# Plot the predictions
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.title.set_text('Predictions')
ax.plot(results[0].predict_result.initial_x, results[0].predict_result.initial_y, label='Initial', c='r')
ax.plot(results[0].predict_result.exact_x, results[0].predict_result.exact_y, label='Exact', c='y')
for i, result in enumerate(results):
    ax.scatter(result.predict_result.predict_x, result.predict_result.predict_y, label=models[i], marker=markers[i])
ax.legend(loc='upper right')
ax.set_xlabel('X')
ax.set_ylabel('Sin(X)')
ax.grid()
plt.savefig('predictions.png')
plt.close()