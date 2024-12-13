import os
from openai import OpenAI
import dotenv
from PIL import Image
import sys
import requests

# load enviromet
dotenv.load_dotenv()

client = OpenAI()

model = "dall-e-3"
size = "1024x1024"
n = 1
prompt = (" ").join(sys.argv[1:])

# Image moderation
if client.moderations.create(input=prompt).results[0]:
    print("Prompt contains disallowed content. Aborting image generation")
    sys.exit(1)


img = client.images.generate(model=model, prompt=prompt, size=size, n=n)
img_path = "test.png"

image_url = img["data"][0]["url"]  # extract image URL from response
generated_image = requests.get(image_url).content  # download the image
image_path = "./"
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)

# Display the image in the default image viewer
image = Image.open(image_path)
image.show()

print(f"Image saved at {img_path}")
