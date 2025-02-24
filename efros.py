import numpy as np
from numpy.typing import NDArray
import util as ut
import time
from multiprocessing import Pool
import efros_util as eu

def find_matches(template: NDArray, sample: NDArray, window_size: int):
    ERR_THRESHOLD = 0.1
    SIGMA = window_size / 6.4
    valid_mask = eu.create_valid_mask(template)
    gauss_mask = eu.gaussian_2d(window_size, SIGMA)
    if gauss_mask.shape != valid_mask.shape:
        raise ValueError(f"Gaussian kernel and mask must have the same shape {gauss_mask.shape} != {valid_mask.shape}")

    ssd = np.full((sample.shape[0], sample.shape[1]), np.inf)
    half_window = window_size // 2
    total_weight = eu.get_total_weight(gauss_mask, valid_mask)
    for i in range(half_window, sample.shape[0] - half_window):
        for j in range(half_window, sample.shape[1] - half_window):
            sample_patch = sample[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            diff = (template - sample_patch) * valid_mask[:, :, np.newaxis]
            ssd[i, j] = np.sum(gauss_mask * np.sum(diff ** 2, axis=-1))
            ssd[i, j] /= total_weight
    matches = []
    min_ssd = np.min(ssd)
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            if ssd[i, j] <= min_ssd * (1 + ERR_THRESHOLD):
                matches.append((sample[i, j], ssd[i, j] - min_ssd))
    return matches


def grow_image(sample_image: NDArray, image: NDArray, window_size: int, max_err_threshold: float):
    visited = eu.init_visited(image)
    times = []
    while len(visited) < image.shape[0] * image.shape[1]:
        start = time.perf_counter()
        progress = False
        pixel_list = eu.get_unfilled_neighbors(image)
        # In parallel, process each unfilled neighbor to find the best match
        with Pool() as pool:
            results = pool.starmap(process_pixel, [(sample_image, image, pixel, window_size, max_err_threshold) for pixel in pixel_list])
        # Update the image with the best matches
        for result in results:
            pixel, best_match, best_match_error = result
            if best_match_error < max_err_threshold:
                image[pixel[0], pixel[1]] = best_match
                visited.add(pixel)
                progress = True
        # If no matches are found for any of the unfilled neighbors, increase the error threshold
        if not progress:
            max_err_threshold *= 1.1
        times.append(time.perf_counter() - start)
        ut.print_report(image, visited, times, max_err_threshold)
        ut.output_image(image, 'results/output.png')
    return image


def process_pixel(sample_image, image, pixel, window_size, max_err_threshold):
    # Template is the window in the new image centered at the pixel
    template = eu.get_neighborhood_window(image, pixel, window_size)
    # Find the best matches in the sample with the current template
    best_matches = find_matches(template, sample_image, window_size)
    if len(best_matches) == 0:
        raise Exception("No matches found")
    # Randomly select one of the best matches
    best_match, best_match_error = best_matches[np.random.randint(len(best_matches))]
    return pixel, best_match, best_match_error
