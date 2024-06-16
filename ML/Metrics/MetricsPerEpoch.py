"""Plots the metrics per epoch"""

import json
from src.graphics import plot_metric_per_epoch


def load_json(json_file_path):
    """Loads the json file and returns it as a dictionary"""
    file_ = open(json_file_path, "r")
    return json.load(file_)


if __name__ == "__main__":
    # loading models
    model_files = {
        "point_net": "../ModelFiles/PointNet.json",
        "mlp": "../ModelFiles/MLP.json",
        "particle_cloud": "../ModelFiles/ParticleCloud.json",
        "combined": "../ModelFiles/CombinedModel.json"
    }
    # metrics for each of the models
    all_metrics = {model: load_json(model_file) for model, model_file in model_files.items()}

    # desired metric
    metric = "loss"
    dict_metrics = {}
    for model_name in all_metrics:
        dict_metrics[model_name] = all_metrics[model_name][metric]
        dict_metrics[f"val_{model_name}"] = all_metrics[model_name][f"val_{metric}"]

    # color dict
    color_dict = {
        "val_mlp": "#01204E",
        "mlp": "#01204E",
        "point_net": "#E9C874",
        "val_point_net": "#E9C874",
        "particle_cloud": "#A34343",
        "val_particle_cloud": "#A34343",
        "combined": "darkgray",
        "val_combined": "darkgray"
    }

    labels_dict = {
        "val_mlp": f"MLP Validation",
        "mlp": f"MLP Trainning",
        "point_net": f"Point-Net Trainning",
        "val_point_net": f"Point-Net Validation",
        "particle_cloud": f"Particle Cloud Trainning",
        "val_particle_cloud": f"Particle Cloud Validation",
        "combined": f"Point-Net + Particle Cloud Trainning",
        "val_combined": f"Point-Net + Particle Cloud Validation"
    }

    plot_metric_per_epoch(
        metrics=dict_metrics,
        labels=labels_dict,
        colors=color_dict,
        metric_name="Loss",
        file_path="../../Plots/LossPerEpoch.pdf"
    )