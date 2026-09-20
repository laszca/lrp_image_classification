"""
Classifies every image in the images/ folder with LRPModel,
computes an LRP relevance map for each, and saves everything to
explanations.pkl for later use by view.py or evaluate_pixel_flipping.py.
"""

from lrp_model import LRPModel
import os
from PIL import Image
import pickle

IMAGES_PATH = "images"


def load_images(images_path="images"):
    """
    Loads every .jpeg image from a directory.

    Args:
        str: images_path
            The path to the directory containing the images.

    Returns:
        list[dict]: One entry per image, containing:
            path (str): The path to the image file.
            original_image (PIL.Image): The loaded image, in RGB.
    """

    # load all images into a list of their paths
    image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpeg')])
    
    # return a dict containing each path and corresponding image
    return [{"path": p, "original_image": Image.open(p).convert("RGB")} for p in image_paths]        
    

def main():

    # load images and model
    images_data = load_images()
    model = LRPModel()
    
    results = []
    
    for image in images_data:
        
        # preprocess image
        image_tensor = model.preprocess(image["original_image"])
        
        # predict classes
        prediction = model.predict(image_tensor)
        
        # use LRP to compute relevances
        relevance_map = model.explain(image_tensor, prediction["predicted_class_index"])
        
        # save results 
        results.append({
            "path": image["path"],
            "original_image": image["original_image"],
            "image_tensor": image_tensor,
            **prediction,
            "relevance_map": relevance_map,
        })
    
    # save results to a pickle file
    with open("explanations.pkl", "wb") as f:
        pickle.dump(results, f)
        
if __name__ == "__main__":
    main()
