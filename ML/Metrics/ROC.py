"""Evaluates the ROC curve and the AUC score for all the trainned models"""
import numpy as np
from sklearn.metrics import roc_curve, auc
from src.utilities import load_data_for_mlp, load_data_as_set_of_particles
import keras
from src.Models.MLP import MLP
from src.Layers.PointNetLayer import PointNetLayer
from src.Models.PointNet import PointNet
from src.Models.ParticleCloud import ParticleCloud
from src.Layers.EdgeConvLayer import EdgeConvLayer
from src.graphics import plot_roc_curve


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the X_test data and returns the signal efficiency (recall) and
    the background rejection (FPR).
    """
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_pred[:, 0])
    return tpr, fpr


if __name__ == "__main__":
    # loading the models

    # MLP
    mlp_model = keras.models.load_model("../ModelFiles/MLP.keras")
    y_test_mlp, X_test_mlp, _ = load_data_for_mlp("../../Data/HiggsTest.csv")
    signal_eff_mlp, back_miss_id_rate = evaluate_model(mlp_model, X_test_mlp, y_test_mlp)

    # PointNet
    custom_objects = {'PointNetLayer': PointNetLayer, 'PointNet': PointNet, 'MLP': MLP}
    point_net_model = keras.models.load_model("../ModelFiles/PointNet.keras", custom_objects=custom_objects)
    X_test, y_test = load_data_as_set_of_particles("../../Data/HiggsTest.csv")
    signal_eff_pn, fpr_pn = evaluate_model(point_net_model, X_test, y_test)

    # Particle Cloud
    custom_objects_pc = {'ParticleCloud': ParticleCloud, 'EdgeConvLayer': EdgeConvLayer, 'MLP': MLP}
    particle_cloud = keras.models.load_model("../ModelFiles/ParticleCloud.keras", custom_objects=custom_objects_pc)
    signal_eff_pc, fpr_pc = evaluate_model(particle_cloud, X_test, y_test)

    # Combined
    combined = keras.models.load_model("../ModelFiles/CombinedModel.keras",
                                       custom_objects={**custom_objects, **custom_objects})
    signal_eff_combined, fpr_combined = evaluate_model(combined, X_test, y_test)

    signal_efficiencies = {
        "MLP": signal_eff_mlp,
        "PointNet": signal_eff_pn,
        "ParticleCloud": signal_eff_pc,
        "Combined": signal_eff_combined
    }
    background_eff = {
        "MLP": back_miss_id_rate,
        "PointNet": fpr_pn,
        "ParticleCloud":  fpr_pc,
        "Combined": fpr_combined
    }
    labels = {
        "MLP": f"Multi-Layer Perceptron (AUC = {auc(back_miss_id_rate, signal_eff_mlp):.2f})",
        "PointNet": f"Point-Net (AUC = {auc(fpr_pn, signal_eff_pn):.2f})",
        "ParticleCloud": f"Particle Cloud (AUC = {auc(fpr_pc, signal_eff_pc):.2f})",
        "Combined": f"Point-Net + Particle Cloud (AUC = {auc(fpr_combined, signal_eff_combined):.2f})",
    }
    colors = {
        "MLP": "#01204E",
        "PointNet": "#E9C874",
        "ParticleCloud": "#A34343",
        "Combined": "darkgray"
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_eff=background_eff,
        labels=labels,
        colors=colors,
        file_path="../../Plots/ROC.pdf"
    )
