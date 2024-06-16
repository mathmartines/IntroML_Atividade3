"""Plots the invariant mass histogram m_wwbb"""

from src.utilities import load_data_for_mlp, load_data_as_set_of_particles
from src.graphics import invariant_mass_hist
from ML.Metrics.usual_metrics import find_threshold
import numpy as np
import keras
import pandas as pd


def separate_signal_and_background(observable, event_label):
    """
    Separates the signal events from background events and returns the values of the observables
    for the signal and background events.
    """
    signal_indices = np.where(event_label == 1)[0]
    background_indices = np.where(event_label == 0)[0]
    # returns the signal and background values for the observable
    return observable[signal_indices], observable[background_indices]


if __name__ == "__main__":
    # true signal and background
    data = pd.read_csv("../../Data/HiggsTest.csv", header=None).to_numpy()
    y_test, inv_mass = data[:, 0], data[:, -1]
    signal, background = separate_signal_and_background(inv_mass, y_test)

    # performing the prediction and fiding the threshold value for the given background rejection
    model = keras.models.load_model("../ModelFiles/CombinedModel.keras")
    # for the MLP
    # y_test, X_test, _ = load_data_for_mlp("../../Data/HiggsTest.csv")
    # for the other NN
    X_test, y_test = load_data_as_set_of_particles("../../Data/HiggsTest.csv")
    scores = model.predict(X_test)[:, 0]
    threshold = find_threshold(y_test[:, 0], scores, background_rejection=0.9)
    y_pred = scores >= threshold
    model_signal, model_background = separate_signal_and_background(inv_mass, y_pred)

    histograms = {
        "signal": signal,
        "background": background,
        "model_signal": model_signal,
        "model_background": model_background
    }

    colors = {
        "signal": "black",
        "background": "red",
        "model_signal": "black",
        "model_background": "red"
    }

    linestyle = {
        "signal": "solid",
        "background": "solid",
        "model_signal": "dashed",
        "model_background": "dashed"
    }

    labels = {
        "signal": "Signal",
        "background": "Background",
        "model_signal": "Point-Net + Particle Cloud Signal (rej=0.9)",
        "model_background": "Point-Net + Particle Cloud Background (rej=0.9)"
    }

    invariant_mass_hist(
        events=histograms,
        labels=labels,
        colors=colors,
        nbins=40,
        linestyle=linestyle,
        title="Point-Net + Particle Cloud",
        file_path="../../Plots/Mwwbb_PointNet_ParticleCloud.pdf"
    )