from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

import time
import os
import requests

def main(image_size=(224, 224)):
    web_url = "https://www.leagueoflegends.com/en-us/champions/"
    response = requests.get(web_url)

    soup = BeautifulSoup(response.content, "html.parser")

    champion_elements = soup.find_all('a', class_="sc-ccb06989-0 jinBDR sc-d043b2-0 bZMlAb")
    champion_names = []

    for element in champion_elements:
        champion_name = element.find("div", {"data-testid": "card-title"}).text
        champion_names.append(champion_name)

    output_dir = "../data/"
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    for name in champion_names:
        for i in range(0, 100):
            image_url = f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{name}_{i}.jpg"
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))
                image_resized = image.resize(image_size)

                image_path = os.path.join(output_dir, f"{name}_{i}.jpg")
                image_resized.save(image_path, "JPEG")
                total_images += 1

    print(f"Data has {len(champion_names)} champions with {total_images} images !!!")
    print("DONE !!!")

if __name__ == "__main__":
    main()