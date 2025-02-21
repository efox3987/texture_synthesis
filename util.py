import numpy as np
from PIL import Image
from numpy.typing import NDArray

def print_report(image: NDArray, visited: set, times):
    print(f"Progress: {((len(visited)/(image.shape[0] * image.shape[1])) * 100):.2f}%")
    seconds = np.mean(times) * (image.shape[0] * image.shape[1] - len(visited))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Estimated time remaining: {int(hours)}:{int(minutes)}:{int(seconds)}")
    print("=========================================")

"""
    Function to create a new image using a seed from the original image
    :param image: PIL image object
    :return: numpy array representing the new image and the original image data
    :const: HSS: seed size, SCALE: SCALE factor
"""
def create_new_image(image, scale):
    # SCALE factor
    new_im_width = int(scale * image.size[0])
    new_im_height = int(scale * image.size[1])

    # Create a new image with the new dimensions
    new_im = np.zeros((new_im_height, new_im_width, 3), dtype=np.uint8)

    # Copy a random seed from original image into the center of the new image
    SEED_SIZE = 10 # seed size, should be even
    HSS = SEED_SIZE // 2

    # Randomly select a seed from the original image
    seed_x = np.random.randint(HSS, image.size[0] - HSS)
    seed_y = np.random.randint(HSS, image.size[1] - HSS)

    # Convert the image to a numpy array
    image_data = np.array(image)

    # Copy the seed into the center of the new image
    midpoint = (int(new_im_height/2), int(new_im_width/2))
    new_im[midpoint[0]-HSS:midpoint[0]+HSS, midpoint[1]-HSS:midpoint[1]+HSS] = image_data[seed_y-HSS:seed_y+HSS, seed_x-HSS:seed_x+HSS]
    return new_im, image_data

def output_image(image, filename):
    Image.fromarray(image).save(filename)
