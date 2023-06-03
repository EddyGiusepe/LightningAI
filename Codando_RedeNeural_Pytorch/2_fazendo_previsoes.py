"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Com este script fazemos as previsões correspondentes.
Lembra que as Images que processamos tem formato 28x28.
"""
import torch
from PIL import Image
from torch import nn, load
from torchvision.transforms import ToTensor



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

"""Instanciamos a nossa Rede Neural"""
clf = ImageClassifier().to('cuda') # ou 'cpu



if __name__ == "__main__":
    with open('Model_RedeNeural_Classification.pt', 'rb') as f:
        clf.load_state_dict(load(f))  

    img = Image.open('img_1.jpg') # Aqui você carrega a Imagem da qual quer fazer a previsão. Lembra que tem que ter 28x28.
    # Dimensões de uma Imagem:
    largura, altura = img.size
    print(f"A imagem que estamos carregando tem formato de: {largura}x{altura}")
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda') # unsqueeze(0) --> Adiciona uma dimensão extra para corresponder à expectativa do Modelo.

    print(torch.argmax(clf(img_tensor))) # Está tentando encontrar o índice do valor máx. retornado pela chamada do Modelo 'clf'.
