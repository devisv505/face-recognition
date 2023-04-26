import requests
from bs4 import BeautifulSoup
from PIL import Image
import face_recognition

url = 'https://edition.cnn.com/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print("start loading...")
for img in soup.find_all('img'):
    try:
        img_url = img.get('src')
        response = requests.get(img_url)
        img = Image.open(io.BytesIO(response.content))
        img = img.convert("RGB")
        img_file = img_url.split('/')[-1] + ".jpg"
        img.save(img_file, "JPEG")
        image = face_recognition.load_image_file(img_file)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            with open(img_file, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f'An error occurred: {e}')
