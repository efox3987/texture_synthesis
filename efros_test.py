import unittest
import numpy as np
from numpy.typing import NDArray
import efros as ef

class TestEfros(unittest.TestCase):

    def test_template_shape(self):
        template = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        self.assertEqual(template.shape, (2, 2, 3))

    def test_get_total_weight(self):
        gauss = np.array([[1, 2], [3, 4]])
        mask = np.array([[1, 0], [0, 1]])
        expected_weight = 5  # 1 + 4
        self.assertEqual(ef.get_total_weight(gauss, mask), expected_weight)

    def test_gaussian_2d(self):
        window_size = 5
        sigma = 1.0
        kernel = ef.gaussian_2d(window_size, sigma)
        self.assertEqual(kernel.shape, (window_size, window_size))
        self.assertAlmostEqual(np.sum(kernel), 1.0, places=5)

    def test_create_valid_mask(self):
        template = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        expected_mask = np.array([[0, 1], [0, 1]], dtype=np.bool_)
        np.testing.assert_array_equal(ef.create_valid_mask(template), expected_mask)

    def test_find_matches(self):
        template = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        sample = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        window_size = 3
        matches = ef.find_matches(template, sample, window_size)
        self.assertTrue(len(matches) > 0)

    def test_get_neighborhood_window(self):
        image = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                          [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                          [[6, 6, 6], [7, 7, 7], [8, 8, 8]]])
        pixel = (1, 1)
        window_size = 3
        expected_window = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                                    [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                                    [[6, 6, 6], [7, 7, 7], [8, 8, 8]]])
        np.testing.assert_array_equal(ef.get_neighborhood_window(image, pixel, window_size), expected_window)

    def test_is_colored(self):
        image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        self.assertTrue(ef.is_colored(image, (0, 1)))
        self.assertFalse(ef.is_colored(image, (0, 0)))

    def test_check_neighbors(self):
        image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        pixel = (1, 1)
        expected_neighbors = [(0, 1), (1, 0)]
        self.assertEqual(ef.check_neighbors(image, pixel), expected_neighbors)

    def test_get_unfilled_neighbors(self):
        image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        expected_neighbors = [(1, 0), (0, 0)]
        self.assertEqual(ef.get_unfilled_neighbors(image), expected_neighbors)

    def test_init_visited(self):
        image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        expected_visited = {(0, 1), (1, 1)}
        self.assertEqual(ef.init_visited(image), expected_visited)

    def test_grow_image(self):
        sample_image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        image = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]])
        window_size = 3
        grown_image = ef.grow_image(sample_image, image, window_size)
        self.assertEqual(grown_image.shape, image.shape)

    def test_process_pixel(self):
        sample_image = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]])
        image = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]])
        pixel = (1, 1)
        window_size = 3
        max_err_threshold = 0.3
        result = ef.process_pixel(sample_image, image, pixel, window_size, max_err_threshold)
        self.assertEqual(len(result), 4)

if __name__ == '__main__':
    unittest.main()
