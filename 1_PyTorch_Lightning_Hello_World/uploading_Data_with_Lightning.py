#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Carregando dados com PyTorch Lightning
======================================
* For pip users:
  --------------
  pip install lightning

* For conda users:
  ----------------
  conda install lightning -c conda-forge
"""
import lightning as L
import pandas as pd

class FilmesDataModule(L.LightningDataModule):
    """Módulo de dados para carregar e processar dados de filmes.
    
    Esta classe herda de LightningDataModule e é responsável por gerenciar
    o carregamento e visualização dos dados de filmes a partir de um arquivo CSV.
    """
    def __init__(self, csv_path: str) -> None:
        """Inicializa o módulo de dados.
    
        Args:
            csv_path: Caminho para o arquivo CSV contendo os dados dos filmes.
        """
        super().__init__()
        self.csv_path = csv_path

    def visualize_data(self):
        """Carrega e visualiza os dados do arquivo CSV.
        
        Exibe as primeiras linhas do DataFrame e informações sobre sua estrutura.
        """
        self.df = pd.read_csv(self.csv_path)
        print(self.df.head())
        print("-"*50)
        self.df.info()
        print("-"*50)
        print(self.df.shape)



if __name__ == "__main__":
    filmes_data = FilmesDataModule(csv_path="n_movies.csv")
    print("")
    filmes_data.visualize_data()
