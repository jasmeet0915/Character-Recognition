import requests
# If you are using a Jupyter notebook, uncomment the following line.
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import json
from PIL import Image
from io import BytesIO

# Add your Computer Vision subscription key and endpoint to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    os.sys.exit()

if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']

ocr_url = endpoint + "vision/v2.1/ocr"

# Set image_url to the URL of an image that you want to analyze.
image_url = "https://scontent-iad3-1.cdninstagram.com/vp/5705232ce1c638788dae3ad50f241eb5/5E287BF8/t51.2885-19/71168521_463626674496299_7067732722701041664_n.jpg?_nc_ht=scontent-iad3-1.cdninstagram.com"


with open("medication.json") as f:
    medicine_data = json.load(f)

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
#image_path = "Medication.jpg"
# Read the image into a byte array
#image_data = open(image_path, "rb").read()
#headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
params = {'language': 'unk', 'detectOrientation': 'true'}
data = {'url': image_url}
response = requests.post(ocr_url, headers=headers, params=params, json=data)
response.raise_for_status()

analysis = response.json()

# Extract the word bounding boxes and text.
line_infos = [region["lines"] for region in analysis["regions"]]
word_infos = []
for line in line_infos:
    for word_metadata in line:
        for word_info in word_metadata["words"]:
            word_infos.append(word_info)
word_infos

# Display the image and overlay it with the extracted text.
plt.figure(figsize=(5, 5))
image = Image.open(BytesIO(requests.get(image_url).content))
ax = plt.imshow(image, alpha=0.5)
for word in word_infos:
    bbox = [int(num) for num in word["boundingBox"].split(",")]
    text = word["text"]
    origin = (bbox[0], bbox[1])
    patch = Rectangle(origin, bbox[2], bbox[3],
                      fill=False, linewidth=2, color='y')
    ax.axes.add_patch(patch)
    plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")
    if text == "TABLET" or text == "CAPSULE":
        y = bbox[1]
        for words in word_infos:
            bbox2 = [int(num) for num in words["boundingBox"].split(",")]
            if bbox2 == y:
                print(words["text"])
plt.show()
plt.axis("off")
print(word_infos)
print(word_infos[0])
print(word_infos[0]['boundingBox'])
