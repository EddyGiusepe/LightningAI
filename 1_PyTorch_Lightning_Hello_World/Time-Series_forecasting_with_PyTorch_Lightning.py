#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Time-Series forecasting with PyTorch Lightning
==============================================
Link de estudo ---> https://lightning.ai/lightning-ai/studios?view=public&section=featured&query=pytorch+lightning

* For pip users:
  --------------
  pip install lightning
* For conda users:
  ----------------
  conda install lightning -c conda-forge
"""
import torch
import lightning as L
import pandas_datareader.data as web # Permite acessar dados financeiros 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger  # Adicione esta importação no topo do arquivo


# Lightning Module for LSTM Model:
class TimeSeriesModel(L.LightningModule):
    def __init__(self, input_dim=1, hidden_dim=150, output_dim=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True) # Long Short-Term Memory
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x) # Processa a sequência através do LSTM
        return self.fc(hidden[-1]) # Aplica a camada fully connected ao último estado oculto para gerar a previsão

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.unsqueeze(-1))
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.unsqueeze(-1))
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


# Custom dataset for time-series data
class StockDataset(Dataset):
    def __init__(self, seq_length=5):
        data = web.DataReader('^DJI', 'stooq')["Close"].values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length], dtype=torch.float),
            torch.tensor(self.data[index+self.seq_length], dtype=torch.float),
        )
    

if __name__ == "__main__":
    # Create DataLoader:
    dataset = StockDataset(seq_length=10)
    # Dividir os dados em treino e validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

    # Initialize model:
    model = TimeSeriesModel()

    # Criar o logger explicitamente
    logger = TensorBoardLogger("./lightning_logs", name="time_series_model")

    # Train model:
    trainer = L.Trainer(
        max_epochs=200,
        logger=True,  # Ativa o logger
        enable_checkpointing=True  # Ativa o salvamento de checkpoints
    )

    trainer.fit(model, train_loader, val_loader)

# Executar no terminal, para usar TensorBoard e visualizar os resultados: 
# tensorboard --logdir=lightning_logs/
