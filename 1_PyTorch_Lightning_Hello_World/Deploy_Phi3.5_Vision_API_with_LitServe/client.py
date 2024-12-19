import base64
import requests
import time

# encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode both images
base64_image1 = encode_image("image1.jpg")
base64_image2 = encode_image("image2.jpg")


# OpenAI based API
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"What are these images?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image1}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image2}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 50,
    "temperature": 0.2
}


t0 = time.time()
response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
print(response.json()["choices"][0])
t1 = time.time()
print(t1-t0)
