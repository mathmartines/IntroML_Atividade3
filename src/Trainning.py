import torch


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct, train_loss = 0, 0
    # performing the trainning over the batches
    for batch, (X, y) in enumerate(dataloader):
        # compute the prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += ((pred > 0.5) == y).all(axis=1).sum().item()
        train_loss += loss.item()

    correct /= size
    train_loss /= len(dataloader)  # number of batches
    print(f"Train set: \n Accuracy: {correct:.4f}, Loss: {train_loss:.4f}")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            model_output = model(X)
            test_loss += loss_fn(model_output, y).item()
            # calculating the correct output
            pred = model_output >= 0.5
            correct += (pred == y).all(axis=1).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Validation Set: \n Accuracy: {correct :.4f}, Avg loss: {test_loss:.4f} \n")