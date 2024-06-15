from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from src.PyTorch.HiggsDataset import HiggsDataset
from torch.utils.data import DataLoader


def display_metrics(y_true, y_pred):
    """Display performance metrics"""
    print(f"Recall: {recall_score(y_true[:, 0], y_pred[:, 0] >= 0.5):.4f}")
    print(f"Precision: {precision_score(y_true[:, 0], y_pred[:, 0] >= 0.5):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true[:, 0], y_pred[:, 0] > 0.5, labels=[0, 1]))


def display_roc_curve(y_true, y_pred):
    """Display ROC curve and the AUC metric"""
    fpr, tpr, thresholds = roc_curve(y_true[:, 0], y_pred[:, 0])
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR (Recall)')
    plt.show()
    print(f"AUC: {auc(fpr, tpr):.4f}")


def plot_hist_trainning(history):
    """Plot the trainning history."""
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca()


def save_model(model, history, model_name):
    """Saves the model and the history"""
    # saving the model
    model.save(f"{model_name}.keras")
    # saving the history
    with open(f"{model_name}.json", "w") as json_file:
        json.dump(history.history, json_file, indent=4)


def load_data_for_mlp(filename):
    """Loads the data for the ModelFiles model"""
    data = pd.read_csv(filename, header=None).to_numpy()
    y, X = data[:, 0], data[:, 1:22]
    y = np.array([[i == label for i in range(2)] for label in y], dtype=np.float32)
    # returning the labels, the data, and all the high-level features
    return y, X, data[:, 22:]


def load_data_as_set_of_particles(filename):
    """Loads the data for the ParticleCloud and PointNet NN"""
    data = HiggsDataset(filename, "cpu")
    # converting from torch tensor to numpy arrays
    data_loader = DataLoader(data, batch_size=len(data))
    # returns the features and the labels
    return map(lambda torch_data: torch_data.numpy(), next(iter(data_loader)))
