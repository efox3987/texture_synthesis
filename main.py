from PIL import Image
import util as ut
import efros as ef

WINDOW_SIZE = 7 # must be odd
SCALE = 1.2
MAX_ERR_THRESHOLD = 0.3
IMAGE = 'nrts/wicker.png'

def main():
    image = Image.open(IMAGE)
    # Create a new image using a seed from the original image
    new_im, old_im = ut.create_new_image(image, SCALE)
    # Grow the new image

    new_im = ef.grow_image(old_im, new_im, WINDOW_SIZE, MAX_ERR_THRESHOLD)
    ut.output_image(new_im, 'results/output.png')

if __name__ == '__main__':
    main()
