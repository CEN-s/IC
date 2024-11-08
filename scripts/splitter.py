from numpy.lib.stride_tricks import as_strided
import numpy as np

def windowed_view(img: np.ndarray, window_shape: tuple[int, int]):
    out_shape = (img.shape[0] - window_shape[0] + 1, img.shape[1] - window_shape[1] + 1) + window_shape
    strides = img.strides * 2
    windows = as_strided(img, out_shape, strides)
    return windows.reshape((windows.shape[0] * windows.shape[1], window_shape[0], window_shape[1]))

class WindowSplitter():
    
    def split(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
            Splits the given image into smaller windows of specified size, with optional padding.
    
            This function divides an image into smaller square windows, each of the size specified
            by the 'window_size' keyword argument. If the 'padding' keyword argument is set to True,
            the image is padded on all sides to ensure that every pixel can be the center of a window.
            The function returns these windows as a reshaped NumPy array.
     
            Args:
                image (np.ndarray): The input image to be split into windows. This should be a 2D array.
                **kwargs: Keyword arguments, where:
                    - 'window_size' (int): The size of each square window. Must be specified.
                    - 'padding' (bool, optional): If True, pads the image before splitting. Defaults to False.
    
            Raises:
                ValueError: If 'window_size' is not provided in the keyword arguments.
    
            Returns:
                np.ndarray: A reshaped NumPy array where each row represents a flattened window of the
                            original image. The number of rows corresponds to the number of windows,
                            and each row has a length of `window_size`**2.
        """
        # Checks if the window size has not been specified.
        if 'window_size' not in kwargs:
            raise ValueError('`window_size` parameter must be specified in WindowSplitter.')
        window_size: int = kwargs['window_size']
    
        # Checks if the padding has been required.
        if 'padding' in kwargs and kwargs['padding'] is True:
            # Calculates the required padding to add to the image borders and apply padding.
            padding = window_size // 2
            image = np.pad(image, padding, mode='edge')
    
        # Extract all possible windows from the image. By default, it excludes border pixels,
        # ensuring each window's center lies entirely within the image's original boundaries.
        # This process guarantees that each extracted window is fully contained within the image.
        #
        # If padding is applied (when 'padding' is set to True), each pixel of the image,
        # including those on the borders, becomes the center of a window. This results in
        # capturing every possible window from the padded image, extending the coverage to
        # the image's edges.
        windows = windowed_view(image, (window_size, window_size))
    
        # Reshape the 'windows' array for output. Each window is flattened into a 1D array.
        # The resulting array shape will have as many rows as there are windows, and each row
        # contains the flattened pixel values of a window. The number of columns in each row
        # is the square of 'windows.shape[1]', which represents the total number of pixels in each window.
        flattened = windows.reshape((windows.shape[0], windows.shape[1] ** 2))
    
        assert flattened.shape[0] == ((image.shape[0] - 2 * (window_size // 2)) * (image.shape[1] - 2 * (window_size // 2))) and flattened.shape[1] == (window_size ** 2), 'Unexpected window array shape.'
    
        return flattened.transpose()