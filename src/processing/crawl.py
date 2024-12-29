from PIL import Image
from io import BytesIO

import os
import requests

def get_schoma_skin(output_dir="../../data/"):
    os.makedirs(output_dir, exist_ok=True)
    total_images = 0

    for name in range(887, 951):
        folder_url = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-chroma-images/{name}/"
        folder_response = requests.get(folder_url)

        if folder_response.status_code == 200:
            for i in range(0, 101):
                index = str(i).zfill(3)
                index = str(name) + index

                image_url = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-chroma-images/{name}/{index}.png"
                image_response = requests.get(image_url)

                if image_response.status_code == 200:
                    image = Image.open(BytesIO(image_response.content))

                    image_path = os.path.join(output_dir, f"{name}_{i}.png")
                    image.save(image_path, "PNG")
                    total_images += 1

    print(f"Has {total_images} schoma images !!!")
    print("DONE !!!")


if __name__ == "__main__":
    get_schoma_skin()