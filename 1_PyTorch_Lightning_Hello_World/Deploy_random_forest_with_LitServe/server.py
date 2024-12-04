#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Install litserve do PyTorch Lightning
=====================================

pip install litserve
"""
import pickle, numpy as np
import litserve as ls # Framework do PyTorch Lightning para servir modelos ML
import os
import logging
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "./model.pkl"

class RandomForestAPI(ls.LitAPI):
    def setup(self, device): # litserve espera o parâmetro device
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        
        with open(MODEL_PATH, "rb") as f:
            try:
                self.model = pickle.load(f)
            except pickle.UnpicklingError:
                raise ValueError("Erro ao carregar o modelo. Verifique se o arquivo é válido.")

    def decode_request(self, request):
        x = np.asarray(request["input"])
        x = np.expand_dims(x, 0)
        return x

    def predict(self, x):
        logging.info("Realizando previsão para entrada %s", x)
        return self.model.predict(x)

    def encode_response(self, output):
        return {
            "class_idx": int(output)
            }

if __name__ == "__main__":
    api = RandomForestAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
    