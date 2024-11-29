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
    def __init__(self, csv_path, str="n_movies.csv"):
        super().__init__()
        self.csv_path = csv_path

    def visualize_data(self):
        self.df = pd.read_csv(self.csv_path)
        print(self.df.head())
        print("-"*50)
        print(self.df.info())



if __name__ == "__main__":
    filmes_data = FilmesDataModule(csv_path="n_movies.csv")
    filmes_data.prepare_data()
    filmes_data.visualize_data()
