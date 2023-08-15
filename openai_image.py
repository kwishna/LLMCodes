import base64
import openai, os

IMG_DIR='./images'

def generate_image(prompt, img_count):
  return openai.Image.create(
    api_key="openai-api-key",
    n=img_count,
    size='512x512',
    response_format='b64_json'
  )

response = generate_image('realistic image of a colorful snake in a human palm.', 2)

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

for i, img_data in enumerate(response['data']):
    img_bytes = base64.b64decode(img_data)

    with open(f'{IMG_DIR}/image_{i+1}.png', 'wb') as f:
        f.write(img_bytes)

    print('URL for the image is : ', img_data.url)
    print(f'Saved image_{i+1}.png in the "images" directory.')
