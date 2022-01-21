from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from torchvision import transforms
import torch


import re

class ABCPreProcess(ABC):
    """Class which implements preprocessing methods for a given object
    """
    
    @abstractmethod
    def process(self, object_i, **parameters):
        pass
    

class PreProcessImageForTraining(ABCPreProcess):
    
    def process(self, object_i: Image, **parameters) -> torch.FloatTensor:
        """Function that pre-process an image for training.

        Args:
            object_i (Image): [description], 
            parameters:{
                crop:{
                    "size": (int), Expected output size of the crop, for each edge.
                    "scale: Tuple(float,float),  :ower and upper bounds for the random area of the crop, before resizing.
                    "ratio": Tuple(float,float),  Lower and upper bounds for the random aspect ratio of the crop, before resizing.
                }
                "mean": (float),
                "std_dev": (float)
            }
            
        Returns:
            torch.FloatTensor: torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        """
        operations = transforms.Compose([
                transforms.RandomResizedCrop(parameters["crop"]["size"], scale=parameters["crop"]["scale"], ratio=parameters["crop"]["ratio"]), # Crop a random portion of image and resize it to a given size.
                transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip the given image randomly with a given probability.
                transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.  (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                transforms.Normalize(mean=parameters["mean"], std_dev=parameters["std_dev"]),
        ])
        return operations(object_i)
        
    
    
class PreProcessImageForEvaluation(ABCPreProcess):
        
    def process(self, object_i: Image, **parameters) -> torch.FloatTensor:
        """Function that pre-process an image for evaluation.
            Args:
                object_i (Image): [description], 
                parameters:{
                    crop:{
                        "size": (int), Desired output size of the crop, for each edge.
                        "center: (int)  Desired output size of the crop after centering
                    }
                    "mean": (float),
                    "std_dev": (float)
              }
                
            Returns:
                torch.FloatTensor: torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        """
        operations = transforms.Compose([
                transforms.Resize(parameters["crop"]["size"]), 
                transforms.CenterCrop(parameters["crop"]["center"]),  # Crops the given image at the center.
                transforms.ToTensor(),
                transforms.Normalize(mean=parameters["mean"], std_dev=parameters["std_dev"]),
        ])
        return operations(object_i)
    
class PreProcessCaption(ABCPreProcess):
    
    def process(self, caption: str, **parameters) -> torch.tensor:
        """Process a caption for being used in the network

        Args:
            caption (str): The caption to be processed.

        Returns:
            torch.tensor: A tensor 
        """
        tokenized_caption = re.findall("[\\w]+", caption.lower())
        return torch.tensor(tokenized_caption)
    
    
    
class PreProcess():
    ImageForTraining = PreProcessImageForTraining
    ImageForEvaluation = PreProcessImageForEvaluation
    Caption = PreProcessCaption
    
    
# ----------------------------------------------------------------
# How to use 

if __name__ == '__main__':
