import numpy as np
from numpy.typing import NDArray


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


def init_visited(image: NDArray) -> set:
    visited = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_colored(image, (i, j)):
                visited.add((i, j))
    return visited


def get_neighborhood_window(image: NDArray, pixel: tuple[int, int], window_size) -> NDArray:
    #half_window = window_size // 2
    #return image[pixel[0] - half_window:pixel[0] + half_window + 1, pixel[1] - half_window:pixel[1] + half_window + 1]
    half_window = window_size // 2
    padded_image = np.pad(image, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='constant', constant_values=0)
    padded_pixel = (pixel[0] + half_window, pixel[1] + half_window)
    return padded_image[padded_pixel[0] - half_window:padded_pixel[0] + half_window + 1, padded_pixel[1] - half_window:padded_pixel[1] + half_window + 1]

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
