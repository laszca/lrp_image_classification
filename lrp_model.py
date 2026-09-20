"""
Wraps a pretrained VGG16 image classifier and exposes prediction and
Layer-wise Relevance Propagation (LRP) explanations through the LRPModel
class.
"""

import torchvision.models as models
import torch
from zennit.composites import EpsilonGammaBox


class LRPModel:
    """
    Loads a pretrained VGG16 model and bundles everything needed to
    classify images and explain predictions with LRP: the network itself,
    its class labels, and the preprocessing/normalization parameters
    derived from the model weights.

    Attributes:
        torch.nn.Module: net
            The pretrained VGG16 network in evaluation mode.
        list: categories
            The 1000 ImageNet class names, indexed by predicted class index.
        torchvision.transforms.Compose: transform
            The preprocessing pipeline matching the loaded weights (resize,
            crop, tensor conversion, normalization).
        torch.Tensor: mean
            Per-channel normalization mean, shape (1, 3, 1, 1).
        torch.Tensor: std
            Per-channel normalization std, shape (1, 3, 1, 1).
        torch.Tensor: low
            Lowest possible normalized pixel value per channel (what a
            pure black pixel maps to), used as the LRP Box rule's lower
            bound.
        torch.Tensor: high
            Highest possible normalized pixel value per channel (what a
            pure white pixel maps to), used as the LRP Box rule's upper
            bound.
    """

    def __init__(self):
        """
        Loads the pretrained VGG16 weights and network, and derives the
        normalization statistics and LRP Box-rule bounds from them.
        """

        # load pretrained VGG16 model
        self.weights = models.VGG16_Weights.DEFAULT
        self.net = models.vgg16(weights=self.weights)
        self.categories = self.weights.meta["categories"]

        # using the corresponding preproccessing structure of the model
        self.transform = self.weights.transforms()

        # saving mean and std of vgg16 training data
        # for later use in shape: (number of pictures (batch), RGB channels, pixels, pixels)
        mean = torch.tensor(self.transform.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.transform.std).view(1, 3, 1, 1)
        self.mean, self.std = mean, std

        # calculating lowest and highest possible pixel value after normalization
        # -> min/max relevances for LRP
        self.low = (0 - mean) / std
        self.high = (1 - mean) / std

        # evaluation mode - no training
        self.net.eval()

        print(f"Model: {self.net}")

    def preprocess(self, image):
        """
        Applies the model's preprocessing pipeline to a single image.

        Args:
            PIL.Image: image
                The image to preprocess, in RGB.

        Returns:
            torch.Tensor: The preprocessed, unbatched image tensor,
                shape (3, 224, 224).
        """
        return self.transform(image)

    
    def forward(self, batch_tensor):
        """
        Performs a forward pass in the NN.

        Args:
            torch.Tensor: batch_tensor
                The batched tensor ready to forward.
            

        Returns:
            torch.Tensor: The forwarded tensor.
        """
        return self.net(batch_tensor)
    
    def predict(self, image_tensor):
        """
        Classifies a single preprocessed image and returns its top-10
        predictions.

        Args:
            torch.Tensor: image_tensor
                The preprocessed, unbatched image tensor,
                shape (3, 224, 224).

        Returns:
            dict: The prediction results, containing:
                predicted_class (str): Name of the top predicted class.
                predicted_class_index (int): Index of the top predicted
                    class.
                confidence (float): Confidence of the top predicted class.
                top_classes (list[str]): Names of the top 10 predicted
                    classes.
                top_probabilities (np.ndarray): Probabilities of the top
                    10 predicted classes.
        """

        # add a first dimension for torch to handle
        batched_tensor = image_tensor.unsqueeze(0)
        
        # perform forward pass, use softmax
        with torch.no_grad():
            probabilities = torch.softmax(self.forward(batched_tensor), dim=1)
        
        # get top ten predictions
        top_probabilities, top_indices = probabilities.topk(10, dim=1)
        top_probabilities = top_probabilities.squeeze().detach().numpy()
        top_indices = top_indices.squeeze().squeeze().numpy()
        top_classes = [self.categories[i] for i in top_indices]
        
        # save predictions in a dict
        prediction = {
            "predicted_class": top_classes[0],
            "predicted_class_index": top_indices[0],
            "confidence": top_probabilities[0],
            "top_classes": top_classes,
            "top_probabilities": top_probabilities,
        }
        
        return prediction
    
    # LRP - Layer-wise Relevance Propagation 
    #   traces back the contributions of each input feature to the final prediction
    def explain(self, image_tensor, target_index):
        """
        Computes a Layer-wise Relevance Propagation (LRP) relevance map,
        showing how much each pixel contributed to the prediction of the
        target class.

        Args:
            torch.Tensor: image_tensor
                The preprocessed, unbatched image tensor,
                shape (3, 224, 224).
            int: target_index
                The class index to explain the prediction for.

        Returns:
            np.ndarray: The relevance map, shape (224, 224). Higher
                values indicate pixels that contributed more strongly to
                the prediction of the target class.
        """

        # first dimension adding + requiring gradients
        batched_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
        
        target = torch.eye(len(self.categories))[target_index].unsqueeze(0)
                    
        ### TODO try different composite rules for LRP and compare results
        # use EpsilonGammaBox composite for LRP
        # "Box" rule for first layer -> pixel values have a lower bound of -3 and an upper bound of 3 
        # LRP gamma rule for convolutional layers -> positive contributions are amplified
        # epsilon rule for all fully connected layers -> small contributions are filtered out
        
        composite = EpsilonGammaBox(low=self.low, high=self.high)
        
        with composite.context(self.net) as modified_model:
            # perform forward pass with modified model for LRP
            result = modified_model(batched_tensor)
            # compute relevance scores using LRP
            relevance, = torch.autograd.grad(result, batched_tensor, grad_outputs=target)
        
        # compute relevance value
        relevance_map = relevance.abs().sum(dim=1).squeeze().detach().numpy()
        return relevance_map


 