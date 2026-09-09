import torchvision.models as models
import torch
import os
from PIL import Image
from zennit.composites import EpsilonGammaBox

MODEL_WEIGHTS = models.VGG16_Weights.DEFAULT
IMAGES_PATH = "images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW = -3.0
HIGH = 3.0

def load_model():
    
    # load pretrained VGG16 model
    model = models.vgg16(weights=MODEL_WEIGHTS)
    
    # evaluate mode - no training
    model.eval()
    
    model.to(DEVICE)
    
    print(f"Model: {model}")
    print(f"Device: {DEVICE}")
    
    return model

def preprocess_image(images_path=IMAGES_PATH):
    
    # load preprocessing pipeline according to the model weights
    preprocess = MODEL_WEIGHTS.transforms()
    
    # load all images into a list of their paths
    image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpeg')])
    
    # build a list of dictionaries containing the image path, the loaded image, and the preprocessed tensor
    images_data = []
    
    for image_path in image_paths:
        
        # load image 
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        
        images_data.append({
            "path": image_path,
            "original_image": image,
            "tensor": input_tensor,
        })
        
    mean = torch.tensor(preprocess.mean).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor(preprocess.std).view(1, 3, 1, 1).to(DEVICE)
    LOW = (0 - mean) / std
    HIGH = (1 - mean) / std


    return images_data

# LRP - Layer-wise Relevance Propagation 
#   traces back the contributions of each input feature to the final prediction
def predict_and_lrp(model, data):
    
    # get categories from model weights
    categories = MODEL_WEIGHTS.meta["categories"]
    
    for image in data:
        # load preprocessed tensor and add batch dimension
        input_tensor = image["tensor"].unsqueeze(0).to(DEVICE)  
        # enable gradient computation for LRP
        input_tensor.requires_grad = True  
        
        # perform forward pass
        result = model(input_tensor)
        # compute probabilities using softmax
        probabilities = torch.softmax(result, dim=1)
        
        
        # get top ten predictions
        top_probabilities, top_indices = probabilities.topk(10, dim=1)
        top_probabilities = top_probabilities.squeeze().detach().cpu().numpy()
        top_indices = top_indices.squeeze().squeeze().cpu().numpy()
        top_classes = [categories[i] for i in top_indices]
        
        # add results to the image dictionary
        image["predicted_class"] = top_classes[0]
        image["confidence"] = float(top_probabilities[0])
        image["top_classes"] = top_classes
        image["top_probabilities"] = top_probabilities
        
        ### LRP
        
        # define target class
        target = torch.eye(len(categories))[top_indices[0]].unsqueeze(0).to(DEVICE)
        
        ### TODO try different composite rules for LRP and compare results
        # use EpsilonGammaBox composite for LRP
        # "Box" rule for first layer -> pixel values have a lower bound of -3 and an upper bound of 3 
        # LRP gamma rule for earlier layers -> positive contributions are amplified
        # epsilon rule for later layers -> small contributions are filtered out
        composite = EpsilonGammaBox(low=LOW, high=HIGH)
        
        
        with composite.context(model) as modified_model:
            # perform forward pass with modified model for LRP
            result = modified_model(input_tensor)
            # compute relevance scores using LRP
            relevance, = torch.autograd.grad(result, input_tensor, grad_outputs=target)
        
        # 
        image["relevance_map"] = relevance.sum(dim=1).squeeze().detach().cpu().numpy()
        
    return data

            