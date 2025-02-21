
# textureSynthesis/efros.py
import numpy as np
from numpy.typing import NDArray
import util as ut
import time
from multiprocessing import Pool

def get_total_weight(gauss: NDArray, mask: NDArray) -> int:
    if gauss.shape != mask.shape:
        raise ValueError("Gaussian kernel and mask must have the same shape")
    total_weight = 0
    for i in range(gauss.shape[0]):
        for j in range(gauss.shape[1]):
            if mask[i, j]:
                total_weight += gauss[i, j]
    return total_weight


def gaussian_2d(window_size: int, sigma: float) -> NDArray:
    """
        Generate a 2D Gaussian kernel without using np.meshgrid.
    """
    center = window_size // 2
    kernel = np.zeros((window_size, window_size))

    for i in range(window_size):
        for j in range(window_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    return kernel / np.sum(kernel)  # Normalize so the sum is 1


def create_valid_mask(template: NDArray):
    res = np.zeros((template.shape[0], template.shape[1]), dtype=np.float32)
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            if template[i, j, 0] != 0 or template[i, j, 1] != 0 or template[i, j, 2] != 0:
                res[i, j] = 1
    return res


def find_matches(template: NDArray, sample: NDArray, window_size: int):
    ERR_THRESHOLD = 0.1
    SIGMA = window_size / 6.4
    valid_mask = create_valid_mask(template)
    gauss_mask = gaussian_2d(window_size, SIGMA)
    if gauss_mask.shape != valid_mask.shape:
        raise ValueError("Gaussian kernel and mask must have the same shape")

    ssd = np.full((sample.shape[0], sample.shape[1]), np.inf)
    half_window = window_size // 2
    for i in range(half_window, sample.shape[0] - half_window):
        for j in range(half_window, sample.shape[1] - half_window):
            sample_patch = sample[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            diff = (template - sample_patch) * valid_mask[:, :, np.newaxis]
            ssd[i, j] = np.sum(gauss_mask * np.sum(diff ** 2, axis=-1))
    matches = []
    min_ssd = np.min(ssd)
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            if ssd[i, j] <= min_ssd * (1 + ERR_THRESHOLD):
                matches.append((sample[i, j], ssd[i, j] - min_ssd))
    return matches


def get_neighborhood_window(image: NDArray, pixel: tuple[int, int], window_size):
    half_window = window_size // 2
    return image[pixel[0] - half_window:pixel[0] + half_window + 1, pixel[1] - half_window:pixel[1] + half_window + 1]


def is_colored(image: NDArray, pixel: tuple[int, int]):
    return np.any(image[pixel[0], pixel[1]] != 0)


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


def get_unfilled_neighbors(image: NDArray):
    res = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_colored(image, (i, j)):
                res += check_neighbors(image, (i, j))
    return list(set(res))


def init_visited(image: NDArray) -> set:
    visited = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_colored(image, (i, j)):
                visited.add((i, j))
    return visited


def grow_image(sample_image: NDArray, image: NDArray, window_size: int, max_err_threshold: float):
    visited = init_visited(image)
    times = []
    while len(visited) < image.shape[0] * image.shape[1]:
        progress = False
        pixel_list = get_unfilled_neighbors(image)
        start = time.perf_counter()
        with Pool() as pool:
            results = pool.starmap(process_pixel, [(sample_image, image, pixel, window_size, max_err_threshold) for pixel in pixel_list])
        for result in results:
            pixel, best_match, best_match_error = result
            if best_match_error < max_err_threshold:
                image[pixel[0], pixel[1]] = best_match
                visited.add(pixel)
                progress = True
        if not progress:
            max_err_threshold *= 1.1
        times.append(time.perf_counter() - start)
        ut.print_report(image, visited, times)
        ut.output_image(image, 'results/output.png')
    return image


def process_pixel(sample_image, image, pixel, window_size, max_err_threshold):
    template = get_neighborhood_window(image, pixel, window_size)
    best_matches = find_matches(template, sample_image, window_size)
    if len(best_matches) == 0:
        raise Exception("No matches found")
    best_match, best_match_error = best_matches[np.random.randint(len(best_matches))]
    return pixel, best_match, best_match_error
