from PIL import Image
import util as ut
import efros as ef

# Parameters for texture synthesis
WINDOW_SIZE = 7 # must be odd
SCALE = 1
SEED_SIZE = 30
MAX_ERR_THRESHOLD = 0.3
IMAGE = 'nrts/carpet.png'

def main():
    ut.create_results_dir()
    image = Image.open(IMAGE)
    # Create a new image using a seed from the original image
    new_im, old_im = ut.create_new_image(image, SCALE, SEED_SIZE)
    # Grow the new image

    new_im = ef.grow_image(old_im, new_im, WINDOW_SIZE, MAX_ERR_THRESHOLD)
    ut.output_image(new_im, 'results/output.png')

if __name__ == '__main__':
    main()
