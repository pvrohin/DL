import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def main():
    # Load data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Reshape y_train and y_test to column vectors
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create a validation set
    val_fraction = 0.2
    val_size = int(len(X_train) * val_fraction)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    # Define the hyperparameters
    batch_size_values = [16, 32, 64, 128]
    learning_rate_values = [0.001, 0.01, 0.1, 0.2]
    epochs_values = [50, 100, 150, 200]
    alpha_values = [0.001, 0.01, 0.1, 1.0]

    # Initialize variables
    np.random.seed(42)
    best_mse = float('inf')
    best_hyperparameters = None

    # Grid search over hyperparameters
    for batch_size in batch_size_values:
        for learning_rate in learning_rate_values:
            for epochs in epochs_values:
                for alpha in alpha_values:
                    # Initialize weights and bias
                    w = np.zeros((X_train.shape[1], 1))
                    b = 0

                    # Train the model using SGD on the training set
                    for epoch in range(epochs):
                        for i in range(0, len(X_train), batch_size):
                            X_batch = X_train[i:i+batch_size]
                            y_batch = y_train[i:i+batch_size]

                            # Compute predictions
                            y_pred = np.dot(X_batch, w) + b

                            # Compute gradients
                            gradient_w = np.dot(X_batch.T, y_pred - y_batch) / len(X_batch)
                            gradient_b = np.sum(y_pred - y_batch) / len(X_batch)

                            # Update weights with L2 regularization
                            w = w - learning_rate * (gradient_w + alpha * w)
                            b = b - learning_rate * gradient_b

                    print(f"Epoch {epoch+1}/{epochs} completed")
                    
                    # Evaluate on the validation set
                    y_val_pred = np.dot(X_val, w) + b
                    val_mse = mean_squared_error(y_val, y_val_pred)

                    print(f"Validation MSE: {val_mse}")

                    # Update best hyperparameters if the current model performs better on the validation set
                    if val_mse < best_mse:
                        best_mse = val_mse
                        best_hyperparameters = (batch_size, learning_rate, epochs, alpha)

    # Use the best hyperparameters to train the final model on the entire training set
    batch_size, learning_rate, epochs, alpha = best_hyperparameters
    w = np.zeros((X_train.shape[1], 1))
    b = 0
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Compute predictions
            y_pred = np.dot(X_batch, w) + b

            # Compute gradients
            gradient_w = np.dot(X_batch.T, y_pred - y_batch) / len(X_batch)
            gradient_b = np.sum(y_pred - y_batch) / len(X_batch)

            # Update weights with L2 regularization
            w = w - learning_rate * (gradient_w + alpha * w)
            b = b - learning_rate * gradient_b

    # Evaluate on the test set
    y_test_pred = np.dot(X_test, w) + b
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Best Hyperparameters: Batch Size={batch_size}, Learning Rate={learning_rate}, Epochs={epochs}, Alpha={alpha}")
    print(f"Test MSE: {test_mse}")


if __name__ == '__main__':
    main()
