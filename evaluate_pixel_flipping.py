"""
Loads the relevance maps saved by explain.py and evaluates
them with the pixel-flipping method, comparing LRP-based pixel removal
against random pixel removal and plotting the resulting confidence
curves.
"""

import pickle
import matplotlib.pyplot as plt
from pixel_flipping import (
    order_lrp_relevant_pixel_positions,
    get_random_pixel_positions,
    confidence_curve
)

with open("explanations.pkl", "rb") as f:
    images_data = pickle.load(f)


def plot_confidences(num_pixel_values, lrp_confidences, random_confidences, predicted_class, image_path):
    """
    Plots and saves the confidence curves for LRP-based vs. random pixel
    removal for a single image.

    Args:
        list[int]: num_pixel_values
            The numbers of pixels removed.
        list[float]: lrp_confidences
            The model's confidence after removing pixels in LRP order.
        list[float]: random_confidences
            The model's confidence after removing pixels in random order.
        str: predicted_class
            The predicted class.
        str: image_path
            The path to the original image.
    """

    plt.plot(num_pixel_values, lrp_confidences, marker='o', label='LRP-based Pixel Removal')
    plt.plot(num_pixel_values, random_confidences, marker='s', label='Random Pixel Removal')
    plt.xlabel('Number of Pixels Removed')
    plt.ylabel(f'Confidence for {predicted_class}')
    plt.ylim(0, 1.1)
    plt.title(f'Confidence Test: Pixel Flipping')
    plt.legend()
    plt.savefig(f'confidences/confidence_test_{predicted_class}_{image_path.split("/")[-1]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    

def main():
    """
    Computes and plots the LRP-based and random pixel-flipping confidence
    curves for every image in explanations.pkl.
    """

    for image_data in images_data:
            
        lrp_pixel_positions = order_lrp_relevant_pixel_positions(image_data["relevance_map"])
        random_pixel_positions = get_random_pixel_positions(image_data["relevance_map"].shape)

        # approx 50000 pixels in 224x224 images -> number of pixel values changed
        num_pixel_values = [0, 100, 500, 1000, 2000, 5000, 10000]
        
        lrp_confidences = confidence_curve(
            image_data["image_tensor"], lrp_pixel_positions, num_pixel_values, image_data["predicted_class_index"],
            )
        random_confidences = confidence_curve(
            image_data["image_tensor"], random_pixel_positions, num_pixel_values, image_data["predicted_class_index"],
            )

        plot_confidences(num_pixel_values, lrp_confidences, random_confidences, image_data["predicted_class"], image_data["path"])

if __name__ == "__main__":
    main()
