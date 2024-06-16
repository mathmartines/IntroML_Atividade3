"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, recall_score, precision_score, confusion_matrix
from src.utilities import load_data_for_mlp, load_data_as_set_of_particles
import keras
import numpy as np


def find_threshold(y_true, scores, background_rejection):
    """
    Evaluates the model on the X_test data and returns the signal efficiency (recall) and
    the background rejection (FPR).
    """
    # background efficiency required
    background_eff = 1 - background_rejection
    # calculating the roc curve
    fpr, _, thresholds = roc_curve(y_true, scores)
    # finds the threshold value for the given background efficiency
    threshold_index = np.where(fpr >= background_eff)[0][0]
    return thresholds[threshold_index]


if __name__ == "__main__":
    # background_rejection
    background_rej = 0.9

    # performing the prediction and fiding the threshold value for the given background rejection
    model = keras.models.load_model("../ModelFiles/PointNet.keras")
    # for the MLP
    # y_test, X_test, _ = load_data_for_mlp("../../Data/HiggsTest.csv")
    # for the other NN
    X_test, y_test = load_data_as_set_of_particles("../../Data/HiggsTest.csv")
    # making the predictions
    y_pred = model.predict(X_test)[:, 0]
    threshold = find_threshold(y_test[:, 0], y_pred, background_rej)
    print(f"threshold value: {threshold}")
    print(f"Performance metrics for a background rejection of {background_rej}:")
    print(f"Signal Efficiency (Recall): {recall_score(y_test[:, 0], y_pred >= threshold):.2f}")
    print(f"Precision: {precision_score(y_test[:, 0], y_pred >= threshold):.2f}")
    print("Confusion Matrix:")
    conf_mat = confusion_matrix(y_test[:, 0], y_pred >= threshold, labels=[0, 1])
    print(conf_mat)
    print(f"Background Efficiency (should be 1 - background rejection): {conf_mat[0][1] / sum(conf_mat[0]): .1f}")
