"""
Pixel-flipping evaluation for LRP relevance maps: orders pixels by
relevance, flips them to a neutral value, and measures how the model's
confidence changes as increasingly many pixels are removed.
"""

import numpy as np
import torch
from lrp_model import LRPModel


def order_lrp_relevant_pixel_positions(relevance_map):
    """
    Orders all pixel positions of a relevance map by relevance, most
    relevant first.

    Args:
        np.ndarray: relevance_map
            The relevance map to order, shape (H, W).

    Returns:
        list[tuple[int, int]]: The (row, col) pixel positions, sorted
            from most to least relevant.
    """

    # sort and flatten the relevance map -> most relevant pixels first
    sorted_relevance_map = np.argsort(relevance_map.flatten())[::-1]
    
    # bring data back into original shape
    sorted_relevance_map = np.unravel_index(sorted_relevance_map, relevance_map.shape)
    
    # list of row and column indices
    positions = list(zip(sorted_relevance_map[0], sorted_relevance_map[1]))
    
    return positions


def get_random_pixel_positions(shape, seed=42):
    """
    Generates all pixel positions of a given shape in a random order, to
    use as a baseline for comparison against LRP-based pixel removal.

    Args:
        tuple[int, int]: shape
            The (height, width) of the image to generate positions for.
        int: seed
            The seed for the random number generator, for reproducibility.

    Returns:
        list[tuple[int, int]]: The (row, col) pixel positions, in random
            order.
    """

    # create a random number generator
    random_generator = np.random.default_rng(seed=seed)
    
    rows, cols = np.indices(shape)
    # flatten and zip rows and columns to create a list of pixel positions
    positions = list(zip(rows.flatten(), cols.flatten()))

    # randomly shuffle the positions
    random_generator.shuffle(positions)

    return positions


def flip_pixels(model, image_tensor, positions, num_pixels):
    """
    Replaces the given number of pixels at the given positions with a
    neutral grey value, to remove their information from the image.

    Args:
        LRPModel: model
            The model instance, used for its mean/std normalization
            values.
        torch.Tensor: image_tensor
            The preprocessed, unbatched image tensor to modify,
            shape (3, 224, 224).
        list[tuple[int, int]]: positions
            The (row, col) pixel positions to flip, ordered by priority.
        int: num_pixels
            The number of pixels (from the start of positions) to flip.

    Returns:
        torch.Tensor: A copy of image_tensor with the given pixels
            flipped.
    """

    # create a copy of the input tensor to avoid modifying the original
    modified_tensor = image_tensor.clone()
    
    # chose 0.5 as a neutral grey value combined with mean and std in order to lose information in this pixel
    # mean and std are tensors with shape (1, 3, 1, 1) for broadcasting -> RGB channels
    neutral_value = (0.5 - model.mean) / model.std
    
    for row, col in positions[:num_pixels]:
        # set the pixel value to neutral grey
        modified_tensor[:, row, col] = neutral_value.squeeze()
        
    return modified_tensor

def confidence_curve(image_tensor, positions, num_pixel_values, target_class_index):
    """
    Measures how the model's confidence in the target class changes as an
    increasing number of pixels are flipped to a neutral value.

    Args:
        torch.Tensor: image_tensor
            The preprocessed, unbatched image tensor, shape (3, 224, 224).
        list[tuple[int, int]]: positions
            The (row, col) pixel positions to flip, ordered by priority.
        list[int]: num_pixel_values
            The numbers of pixels to flip, evaluated one at a time.
        int: target_class_index
            The class index to track the confidence for.

    Returns:
        list[float]: The model's confidence in the target class, one
            value per entry in num_pixel_values.
    """

    model = LRPModel()
    confidences = []
    
    for num_pixels in num_pixel_values:
        # flip pixels
        modified_tensor = flip_pixels(model, image_tensor, positions, num_pixels)
        
        # add batch dimension
        modified_batch = modified_tensor.unsqueeze(0)
        
        # only forward pass
        with torch.no_grad():
            
            result = model.forward(modified_batch)
            probabilities = torch.softmax(result, dim=1)
            
            # save confidences
            confidence = probabilities[0, target_class_index].item()
            confidences.append(confidence)
        
    return confidences