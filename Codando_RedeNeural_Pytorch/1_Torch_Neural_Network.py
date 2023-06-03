"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Aqui treinamos e fazemos previsões com o um Modelo de Classificação de Imagens.
Este estudo foi baseado no Tutorial de Nicholas Renotte 🤩
"""

"""Instlando as bibliotecas necessárias"""
from torch import nn, save, load # Para usar Classes e construir Redes Neurais
from torch.optim import Adam
from torch.utils.data import DataLoader # Para carregar nossos Dados
from torchvision import datasets
from torchvision.transforms import ToTensor # Para converter as nossas Imagens em Tensores

"""Baixando e carregando nosso Dataset"""
# Vamos a usar os dados de MNIST
train = datasets.MNIST(root="data",
                       download=True,
                       train=True,
                       transform=ToTensor())

dataset = DataLoader(train, 32)

"""
Lembrar: 
* A nossa Imagem é (1, 28, 28) e temos de 0-9 Classes. 
* Vamos trabalhar com nn.Conv2d devido a natureza Bidimensional das Imagens que estamos utilizando.
"""
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__() # Chama o construtor da classe pai para inicializar a Classe base.
        self.model = nn.Sequential(nn.Conv2d(1, 32, (3, 3)), # Apenas um canal (preto e branco), gera 32 canais de saída e usa um filtro de convolução 3x3.
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, (3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3, 3)),
                                   nn.ReLU(),
                                   nn.Flatten(), # Aplana a saída em um vetor Unidimensional. 
                                   nn.Linear(64*(28-6)*(28-6), 10) # Camada totalmente conectada. Esta camada é usada para a Classificação final das 10 classes possíveis.
                                  )


    def forward(self, x):
        return self.model(x)

"""Vamos a instanciar a nossa Rede Neural, Optimizer e a Loss"""
clf = ImageClassifier().to('cuda') # ou 'cpu

opt = Adam(clf.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()


"""Fluxo de Treinamento"""
if __name__ == '__main__':
    for epoch in range(10): # Treinamos para 10 épocas
        for batch in dataset:
            X, y = batch # Aqui descompactamos 
            X, y = X.to('cuda'), y.to('cuda') # Processamos nossos Dados com 'cuda'
            yhat = clf(X) # Fazemos a previsão
            loss = loss_fn(yhat, y) 

            # Aplicamos Backprop
            opt.zero_grad() # Zerar os Gradientes
            loss.backward() # Backpropagation
            opt.step() # Descida de gradiente 

        print(f"Épocas: {epoch} Loss é: {loss.item()}")

    with open('Model_RedeNeural_Classification.pt', 'wb') as f:
        save(clf.state_dict(), f)
        