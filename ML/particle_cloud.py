import torch
from torch import nn
from torch.utils.data import DataLoader
from src.PyTorch.HiggsDataset import HiggsDataset
from src.PyTorch.Trainning import train_loop, test_loop
from src.PyTorch.NN_ParticleCloud import ParticleCloud

if __name__ == "__main__":
    device = "cpu"

    higges_trainning = HiggsDataset("../Data/HiggsTrainning.csv", device)
    higgs_validation = HiggsDataset("../Data/HiggsValidation.csv", device)

    particle_cloud = ParticleCloud().to(device)

    train_data = DataLoader(higges_trainning, batch_size=32)
    validation_data = DataLoader(higgs_validation, batch_size=32)

    # initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(particle_cloud.parameters(), lr=1e-3)

    epochs = 50
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_data, particle_cloud, loss_fn, optimizer)
        test_loop(validation_data, particle_cloud, loss_fn)
    print("Done!")

    # save model
    torch.save(particle_cloud.state_dict(), "particle_cloud.pth")
