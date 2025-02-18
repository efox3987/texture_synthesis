import numpy as np
from numpy.typing import NDArray
import util as ut

"""
    Function to get the total weight of the Gaussian kernel
    :param gauss: numpy array representing the Gaussian kernel
    :param mask: numpy array representing the mask
    :return: int representing the total weight
"""
def get_total_weight(gauss: NDArray, mask: NDArray) -> int:
    if gauss.shape != mask.shape:
        raise ValueError("Gaussian kernel and mask must have the same shape")
    total_weight = 0
    for i in range(0, gauss.shape[0]):
        for j in range(0, gauss.shape[1]):
            if mask[i, j]:
                total_weight += gauss[i, j]
    return total_weight

"""
    Function to create a 2D Gaussian kernel
    :param window_size: int representing the window size
    :param sigma: float representing the sigma value
    :return: numpy array representing the 2D Gaussian kernel
"""
def gaussian_2d(window_size: int, sigma: float) -> np.ndarray:

    ax = np.linspace(-(window_size // 2), window_size // 2, window_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize so the sum is 1


"""
    Function to create a valid mask
    :param template: numpy array representing the template
    :return: numpy array representing the valid mask
"""
def create_valid_mask(template: NDArray):
    res = np.zeros((template.shape[0], template.shape[1]), dtype=np.bool)
    for i in range(0, template.shape[0]):
        for j in range(0, template.shape[1]):
            if template[i, j, 0] != 0 or template[i, j, 1] != 0 or template[i, j, 2] != 0:
                res[i, j] = 1
    return res


"""
    Function to find the best matches
    :param template: numpy array representing the template
    :param sample: numpy array representing the sample
    :param window_size: int representing the window size
    :return: list of tuples representing the best matches
"""
def find_matches(template: NDArray, sample: NDArray, window_size: int):
    ERR_THRESHOLD = 1.1
    SIGMA = window_size / 6.4
    valid_mask = create_valid_mask(template)
    gauss_mask = gaussian_2d(window_size, SIGMA)
    total_weight = get_total_weight(gauss_mask, valid_mask)
    ssd = np.zeros((sample.shape[0], sample.shape[1]), dtype=float)
    for i in range(0, len(sample)):
        for j in range(0, len(sample[0])):
            for ii in range(0, len(template)):
                for jj in range(0, len(template[0])):
                    # Convert pixel to value 0-1 and calculate the distance
                    dist = np.sum((template[ii, jj]/255 - sample[i-ii, j-jj]/255)**2)
                    ssd[i,j] = ssd[i,j] + dist*gauss_mask[ii,jj]*valid_mask[ii,jj]
                ssd[i,j] = ssd[i,j]/total_weight
    matches = []
    for i in range(0, len(sample)):
        for j in range(0, len(sample[0])):
            if ssd[i,j] <= np.min(ssd)*(1+ERR_THRESHOLD):
                matches.append((sample[i,j], ssd[i,j] - np.min(ssd)))
    return matches


"""
    Function to get the neighborhood window of a pixel
"""
def get_neighborhood_window(image: NDArray, pixel: tuple[int, int], window_size):
    window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
    for i in range(0, window_size):
        for j in range(0, window_size):
            window[i, j] = image[pixel[0] - window_size // 2 + i, pixel[1] - window_size // 2 + j]
    return window

"""
    Function to check if a pixel is colored
    :param image: numpy array representing the image
    :param pixel: tuple representing the pixel coordinates
    :return: True if the pixel is colored, False otherwise
"""
def is_colored(image: NDArray, pixel: tuple[int, int]):
    for i in range(0, 3):
        if image[pixel[0], pixel[1], i] != 0:
            return True
    return False


"""
    Function to check the neighbors of a pixel that are colored
    :param image: numpy array representing the image
    :param pixel: tuple representing the pixel coordinates
    :return: list of tuples representing the coordinates of the colored neighbors
"""
def check_neighbors(image: NDArray, pixel: tuple[int, int]):
    res = []
    if pixel[0] > 0 and not is_colored(image, (pixel[0] - 1, pixel[1])):
        res.append((pixel[0] - 1, pixel[1]))
    if pixel[0] < image.shape[0] - 1 and not is_colored(image, (pixel[0] + 1, pixel[1])):
        res.append((pixel[0] + 1, pixel[1]))
    if pixel[1] > 0 and not is_colored(image, (pixel[0], pixel[1] - 1)):
        res.append((pixel[0], pixel[1] - 1))
    if pixel[1] < image.shape[1] - 1 and not is_colored(image, (pixel[0], pixel[1] + 1)):
        res.append((pixel[0], pixel[1] + 1))
    return res


"""
    Function to get the unfilled neighbors of the image
    :param image: numpy array representing the image
    :return: list of tuples representing the coordinates of the unfilled neighbors
"""
def get_unfilled_neighbors(image: NDArray):
    res = []
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if is_colored(image, (i, j)):
                res += check_neighbors(image, (i, j))
    return list(set(res))


"""
    Function to initialize the visited set
    :param image: numpy array representing the image
    :return: set of tuples representing the visited pixels
"""
def init_visited(image: NDArray):
    visited = set()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if is_colored(image, (i, j)):
                visited.add((i, j))
    return visited


"""
    Function to grow the image
    :param sample_image: numpy array representing the sample image
    :param image: numpy array representing the image
    :param window_size: int representing the window size
    :return: numpy array representing the grown image
"""
def grow_image(sample_image: NDArray, image: NDArray, window_size: int):
    visited = init_visited(image)

    max_err_threshold = 0.3

    while len(visited) < image.shape[0] * image.shape[1]:
        progress = False
        pixel_list = get_unfilled_neighbors(image)
        for pixel in pixel_list:
            print(f"Calculating pixel: {pixel}")
            template = get_neighborhood_window(image, pixel, window_size)
            best_matches = find_matches(template, sample_image, window_size)
            if len(best_matches) == 0:
                raise Exception("No matches found")
            best_match, best_match_error = best_matches[np.random.randint(len(best_matches))]
            if best_match_error < max_err_threshold:
                image[pixel[0], pixel[1]] = best_match
                visited.add(pixel)
                progress = True
        if not progress:
            max_err_threshold *= 1.1
        print(f"Progress: {(len(visited)/(image.shape[0] * image.shape[1])) * 100}%")
        ut.output_image(image, 'results/output.png')
    return image
