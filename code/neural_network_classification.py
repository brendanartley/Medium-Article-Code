from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class PredictionPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, print_freq):
        self.X = X
        self.y = y
        self.print_freq = print_freq
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.print_freq==0:
            plot_model_predictions(self.model, self.X, self.y, epoch)  

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def generate_data_sample(n_samples, noise, seed, plot_sample = True):
    # Generates a circlular cluster around a central cluster
    X1, y1 = datasets.make_circles(n_samples=n_samples//2, noise=noise, factor=.99999, random_state=seed)
    y1 = np.ones((len(X1),), dtype=int)
    X2, y2 = datasets.make_blobs(n_samples=n_samples//2, cluster_std=noise*1.5, n_features=1, centers= [(0,0)], random_state=seed)
    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])

    if plot_sample:
        # plot sample w/ Matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        ax1 = plt.subplot(1, 1, 1)
        ax1.scatter(X[y==1][:, 0], X[y==1][:, 1], c='tab:blue', s=25, label=True) # numpy supports boolean indexing
        ax1.scatter(X[y==0][:, 0], X[y==0][:, 1], c='tab:orange', s=25, label=False)
        ax1.set_title("Data Sample")
        ax1.legend()
        plt.show()

    return X, y

def generate_theoretical_data(n, X):
    # Generates even distribution of points in 2D euclidian space
    # ie. np.linspace() for 2D
    x = np.linspace(X[:, 0].min()*1.1, X[:, 0].max()*1.1, n)
    y = np.linspace(X[:, 1].min()*1.1, X[:, 1].max()*1.1, n)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()
    X_theoretical = np.stack((xv, yv), axis=1)
    return X_theoretical

def plot_model_predictions(model, X, y, epoch, last=False):
    # Predictions on actual data
    raw_preds = model(X)
    preds = np.argmax(model(X), 1)

    # Generating predictions on theoretical data
    tx = generate_theoretical_data(200, X)
    tx_preds = np.argmax(model(tx), 1)

    # ---- Matplotlib Plot ----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax1 = plt.subplot(1, 1, 1)

    # plotting theoretical predictions
    ax1.scatter(tx[tx_preds==1][:, 0], tx[tx_preds==1][:, 1], 
                c='tab:blue', s=25, alpha=0.05, edgecolors='none')

    ax1.scatter(tx[tx_preds==0][:, 0], tx[tx_preds==0][:, 1],
                c='tab:orange', s=25, alpha=0.05,edgecolors='none')
    
    # plotting validation predictions
    ax1.scatter(X[preds==1][:, 0], X[preds==1][:, 1], c='tab:blue', s=25, label=True)
    ax1.scatter(X[preds==0][:, 0], X[preds==0][:, 1], c='tab:orange', s=25, label=False)
    
    # Setting plot boundaries to remove whitespace
    ax1.set_xlim([X[:, 0].min()*1.1, X[:, 0].max()*1])
    ax1.set_ylim([X[:, 1].min()*1.1, X[:, 1].max()*1.1])

    # Setting Title
    try:
        ax1.set_title("Log Loss: {:.4f}, Epoch: {}".format(log_loss(y, preds), epoch))
    except:
        ax1.set_title("Error. One Class predicted.")
    
    # Keeping plot open if last epoch
    if last:
        ax1.scatter(X[y!=preds][:, 0], X[y!=preds][:, 1], c='tab:red', s=25, label="Errors")
        ax1.legend(loc='upper right')
        plt.show()
    else:
        ax1.legend(loc='upper right')
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()
    return

def get_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2)
        ])
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Optimizer for learning rate
            loss=tf.keras.losses.BinaryCrossentropy(), # Loss function to minimize
            metrics=[tf.keras.losses.BinaryCrossentropy()], # List of metrics to monitor (can be multiple)
        )
    return model


def main():
    seed = 2718
    set_seeds(seed)

    epochs = 400
    batch_size = 20
    print_freq = 25

    X, y = generate_data_sample(1000, 0.15, seed, plot_sample=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    model = get_compiled_model()
    pp_callback = PredictionPlotCallback(X, y, print_freq)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[pp_callback]
    )

    plot_model_predictions(model, X, y, epochs, last=True)
    
if __name__ == '__main__':
    main()