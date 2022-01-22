from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from torchvision import transforms
import torch
from Dataset import Dataset, DatasetState
from Sample import Sample
from Vocabulary import Vocabulary
import re
from typing import Tuple

class ABCPreProcess(ABC):
    """Class which implements preprocessing methods for a given object
    """
    
    @abstractmethod
    def process(object_i, **parameters):
        pass
    

class PreProcessImageForTraining(ABCPreProcess):
    
    @staticmethod
    def process(object_i: Image, parameters) -> torch.FloatTensor:
        """
        Function that pre-process an image for training.

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
                transforms.Normalize(mean=parameters["mean"], std=parameters["std_dev"]),
        ])
        return operations(object_i)
        
    
    
class PreProcessImageForEvaluation(ABCPreProcess):
    
    @staticmethod
    def process(object_i: Image, **parameters) -> torch.FloatTensor:
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
                transforms.Normalize(mean=parameters["mean"], std=parameters["std_dev"]),
        ])
        return operations(object_i)
    
class PreProcessCaption(ABCPreProcess):
    
    @staticmethod
    def process(caption: str, **parameters) -> list[str]:
        """Process a caption for being used in the network

        Args:
            caption (str): The caption to be processed.

        Returns:
            torch.tensor: A tensor 
        """
        tokenized_caption = re.findall("[\\w]+|\.|\,", caption.lower())
        return tokenized_caption

#TO Do
class PreProcessDatasetForTraining(ABCPreProcess):
    
    image_trasformation_parameter = {
        "crop":{
            "size": 32,
            "scale": (0.08,1.0),
            "ratio": (3. / 4., 4. / 3.),
        },
        "mean": torch.tensor([0.485, 0.456, 0.406]), # the mean of the training data on the 3 channels (RGB)
        "std_dev": torch.tensor([0.229, 0.224, 0.225]) # the standard deviation of the training data on the 3 channels (RGB)
    }
    @staticmethod
    def process(dataset: Dataset, vocabulary: Vocabulary) -> Tuple[Dataset,Vocabulary]:
        
        # Control block
        if dataset.state == DatasetState.Training:
            torch.warnings.warn("The Dataset is already prepared for Training, another pre-process training could lead to some inconsistence.")
            
        if dataset.state == DatasetState.Evaluation:
            torch.warnings.warn("The Dataset is already prepared for Evaluation, pre-process for training could lead to some inconsistence.")
        
        # PreProcess block
        for sample in dataset.dataset["sample"]:
            sample.alter_caption(PreProcess.Caption.process(sample.caption))
            sample.alter_image(PreProcess.ImageForTraining.process(sample.image, PreProcessDatasetForTraining.image_trasformation_parameter))
            
        # Enrich the vocabulary
        vocabulary.make_enrich = True
        vocabulary.bulk_enrich([sample.caption for sample in dataset.dataset["sample"][:]])
        vocabulary.make_enrich = False
        
        return dataset, vocabulary
#TO Do
class PreProcessDatasetForEvaluation(ABCPreProcess):
    
    @staticmethod
    def process(dataset: Dataset, vocabulary: Vocabulary) -> Tuple[Dataset,Vocabulary]:
            pass
    
class PreProcess():
    ImageForTraining = PreProcessImageForTraining
    ImageForEvaluation = PreProcessImageForEvaluation
    Caption = PreProcessCaption
    DatasetForTraining = PreProcessDatasetForTraining
    DatasetForEvaluation = PreProcessDatasetForTraining
    
# ----------------------------------------------------------------
# How to use 

if __name__ == '__main__':
    
    ds = Dataset("./dataset")
    df = ds.get_fraction_of_dataset(percentage=10)
    
    v = Vocabulary(verbose=True)
    # Make a translation
    print(v.translate(["I","like","PLay","piano","."]))
    
    df_pre_processed,v_enriched = PreProcess.DatasetForTraining.process(dataset=df,vocabulary=v)
    print(df_pre_processed)