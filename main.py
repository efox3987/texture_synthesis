from PIL import Image
import numpy as np

# Open the image file
image = Image.open('nrts/carpet.png')

new_im_width = 2 * image.size[0]
new_im_height = 2 * image.size[1]

# Create a new image with the new dimensions
new_im = np.zeros((new_im_height, new_im_width, 3), dtype=np.uint8)

# Convert the image to a numpy array
image_data = np.array(image)

# Copy random 3x3 seed from original image into the center of the new image

seed_size = 5

seed_x = np.random.randint(seed_size, image.size[0] - seed_size)
seed_y = np.random.randint(seed_size, image.size[1] - seed_size)
hss = int(seed_size/2)
midpoint = (int(new_im_height/2), int(new_im_width/2))
new_im[midpoint[0]-hss:midpoint[0]+hss, midpoint[1]-hss:midpoint[1]+hss] = image_data[seed_y:seed_y + seed_size, seed_x:seed_x + seed_size]

# Save the new image
new_im = Image.fromarray(new_im)
new_im.save('results/output.png')
