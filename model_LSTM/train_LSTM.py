import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from dataload import StockDataset
from LSTM import LSTMModel

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the LSTM model.

    Parameters:
        model (nn.Module): LSTM model.
        dataloader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")


def predict(model, dataloader):
    """
    Predict using the trained model.

    Parameters:
        model (nn.Module): Trained LSTM model.
        dataloader (DataLoader): DataLoader for test data.

    Returns:
        list: Predicted values.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs = model(x)
            predictions.extend(outputs.squeeze().cpu().numpy())
    return predictions



if __name__ == "__main__":
    # Parameters
    sequence_length = 50
    input_size = 5
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Load data
    data = pd.read_csv("data_folder_with_features/AAPL.csv", index_col="Date", parse_dates=True)

    # Prepare dataset and dataloader
    dataset = StockDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, dataloader, criterion, optimizer, num_epochs)