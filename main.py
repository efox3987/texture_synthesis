from PIL import Image
import util as ut
import efros as ef


def main():
    image = Image.open('nrts/carpet.png')
    # Create a new image using a seed from the original image
    new_im, old_im = ut.create_new_image(image)
    Image.fromarray(new_im).show()

    # Grow the new image
    WINDOW_SIZE = 5
    new_im = ef.grow_image(old_im, new_im, WINDOW_SIZE)
    ut.output_image(new_im, 'results/output.png')


if __name__ == '__main__':
    main()
