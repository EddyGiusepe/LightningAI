# Exemplo:
# python client.py --image dogs-playing-in-grassy-field.jpg --prompt "O que tem esta imagem?"
import argparse
from openai import OpenAI
from termcolor import colored

from src.config import DEFAULT_MODEL

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1/",
    api_key="ollama",
)


def send_image_for_processing(input_image_path: str, prompt: str):
    """Send an image to the server to generate a caption."""

    # Open and read the input image file
    image_url = f"file://{input_image_path}"

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        stream=True,
        max_tokens=300,
    )
    print(colored("Processing image...", "yellow"))
    for chunk in stream:
        print(colored(chunk.choices[0].delta.content or "", "green"), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Envie uma imagem para o servidor para gerar uma legenda."
    )
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
    parser.add_argument("-p", "--prompt", help="Prompt para a imagem", default="Descreva esta imagem.")

    args = parser.parse_args()
    send_image_for_processing(args.image, args.prompt)
