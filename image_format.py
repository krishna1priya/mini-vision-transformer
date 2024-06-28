from PIL import Image

# Load the image
input_image_path = 'deer.png'
output_image_path = 'test_deer_image.png'

with Image.open(input_image_path) as img:
    # Resize the image to 384x384
    img_resized = img.resize((384, 384))
    img_resized.save(output_image_path, format='PNG')

output_image_path
