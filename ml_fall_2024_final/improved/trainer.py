import numpy as np
import torch
from torch import nn


def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_cont_batch, _, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_cont_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, criterion, scaler):
    mae = nn.L1Loss()

    model.eval()
    test_loss = 0.0
    predictions, targets = [], []
    features = []
    mae_list = []
    mse_list = []
    with torch.no_grad():
        for X_cont_batch, _, y_batch in test_loader:
            features.extend(X_cont_batch[i] for i in range(len(X_cont_batch)))
            outputs = model(X_cont_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            # inverse_transform
            predictions.append(outputs)
            targets.append(y_batch)

    # 转换为数组
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    print(features[0].shape)
    for i in range(len(predictions)):
        t1 = torch.tensor(predictions[i]).reshape(-1, 1)
        t2 = torch.tensor(targets[i]).reshape(-1, 1)
        whole1 = torch.cat((features[i], t1), dim=1)
        whole2 = torch.cat((features[i], t2), dim=1)
        whole1 = scaler.inverse_transform(whole1)
        whole2 = scaler.inverse_transform(whole2)
        predictions[i] = whole1[:, -1]
        targets[i] = whole2[:, -1]
        mae_list.append(mae(torch.tensor(predictions[i]), torch.tensor(targets[i])).item())
        mse_list.append(criterion(torch.tensor(predictions[i]), torch.tensor(targets[i])).item())
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    return predictions, targets, mae_list, mse_list
