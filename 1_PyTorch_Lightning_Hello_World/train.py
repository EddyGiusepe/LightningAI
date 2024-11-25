#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Install PyTorch Lightning
=========================
Link de estudo ---> https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#hello-simple-model

* For pip users:
  --------------
  pip install lightning
* For conda users:
  ----------------
  conda install lightning -c conda-forge
"""
import torch
import lightning as L

# Step 1: Define your Lightning Module
class ToyExample(L.LightningModule):
    def __init__(self, model):
        super().__init__() # Chama o construtor da classe pai (LightningModule)
        self.model = model

    def training_step(self, batch):
        """Envie o lote pelo modelo e calcule a perda.
           O Trainer executará .backward(), optimizer.step(), 
           .zero_grad(), etc. para você
        """
        loss = self.model(batch).sum()
        return loss
    
    def configure_optimizers(self):
        """Escolha um otimizador ou implemente o seu próprio.
        """
        return torch.optim.Adam(self.parameters())


# Step 2: Run the trainer
if __name__ == "__main__":
    # Configure o modelo para que ele possa ser chamado em `training_step`.
    # Este é um modelo fictício. Substitua-o por um LLM ou algo assim
    model = torch.nn.Linear(32, 2)
    pl_module = ToyExample(model)

    # Configure o conjunto de dados e retorne um carregador de dados.
    train_loader = torch.utils.data.DataLoader(torch.rand(8,32))
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(pl_module, train_loader)
