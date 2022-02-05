# Typing trick for avoid circular import dependencies valid for python > 3.9
# from __future__ import annotations
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .Vocabulary import Vocabulary

import os
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
import re
from torchvision import transforms
from VARIABLE import MAX_CAPTION_LENGTH, IMAGES_SUBDIRECTORY_NAME, CAPTION_FILE_NAME
from typing import Tuple, List, Iterable


class MyDataset(Dataset):
    """
        Wrapper of Dataset Pytorch Object.
        For our scopes the dataset folder must follow this rule:
            
            1) As a child of the directory, we must have a csv named `CAPTION_FILE_NAME` that follow this pattern:\n
                `image_name| comment_number| comment`\n
                Example:    1000092795.jpg| 0| Two young guys with shaggy hair look at their hands while hanging out in the yard .
            
            2) As brother of the csv file we must have the folder of the images, the directory name is a variable `IMAGES_SUBDIRECTORY_NAME`
        
        Assumption: 

            1) The dataset will pick only the caption less then the variable `MAX_CAPTION_LENGTH`
            
    """
    image_trasformation_parameter = {
        "crop":{
            "size": 224
        },
        "mean": torch.tensor([0.485, 0.456, 0.406]), # the mean of the training data on the 3 channels (RGB)
        "std_dev": torch.tensor([0.229, 0.224, 0.225]) # the standard deviation of the training data on the 3 channels (RGB)
    }
    
    def __init__(self, directory_of_data:str , percentage:int = 100, already_computed_dataframe: pd.DataFrame = None):
        """Create a new dataset from source files or from a preprocessed dataset.

        Args:
            directory_of_data (str, mandatory): 
                The directory tagged as root for the dataset.
            
            percentage (int, optional): Default is 100.
                The percentage of row that we want store in our object.
                
            already_computed_dataframe (pd.DataFrame, Optional): Default is None.
                If the dataset is computed outside put it there.
                REMARK Please follow the rule:
                    | Index | image_name |  <List(str)> Caption |\n
                    |:-----:|:----------:|:--------------------:|\n
                    |   0   |  pippo.jpg | ["i","like","pizza"] |\n
                    
        Raises:
            ValueError: if the dataset directory is invalid (Not Exist, Not a directory).
        """
        
        # If the constructor receive a dataframe, we assume that it is already manipulated for doing our operation, no further op. needed.
        if already_computed_dataframe is not None:
            self.directory_of_data = directory_of_data
            self._dataset = already_computed_dataframe
            return
        
        # Input checking
        if not os.path.exists(directory_of_data):
            raise ValueError(f"{directory_of_data} not Exist!")
        if not os.path.isdir(directory_of_data):
            raise ValueError(f"{directory_of_data} is not a directory!")
        
        self.directory_of_data = directory_of_data
        
        # Load the dataset
        _temp_dataset: pd.DataFrame = pd.read_csv(f"{directory_of_data}/{CAPTION_FILE_NAME}", sep="|", skipinitialspace=True)[["image_name","comment"]]
        
        # Split every caption in its words. 
        _temp_dataset["comment"] = _temp_dataset["comment"].apply( lambda comment: re.findall("[\\w]+|\.|\,",str(comment).lower()))
        
        # Filter for retrieve only caption with a length less than MAX_CAPTION_LENGTH length
        _temp_dataset = _temp_dataset[ _temp_dataset["comment"].map(len) <= MAX_CAPTION_LENGTH]
        
        # Pick only a given percentage of the row in the dataset
        self._dataset = _temp_dataset.head(int(len(_temp_dataset)*(percentage/100)))
        
    def get_fraction_of_dataset(self, percentage: int, delete_transfered_from_source: bool = False):
        """Get a fraction of the dataset 

        Args:
            percentage (int): 
                The percentage of row that we want store in our new object.
                
            delete_transfered_from_source (bool, optional): Defaults to False.
                Tell if you want to delete the row in the source object that are transfered to the new object.
                
        Returns:
            (MyDataset): 
                The new computed dataset object.
        """
        # Retrieve the number of rows 
        _temp_df_moved: pd.DataFrame = self._dataset.head(int(len(self._dataset)*(percentage/100))).sample(frac=1)
        
        # Deep copy of the dataframe
        _temp_df_copy = _temp_df_moved.copy()
        
        # If delete_transfered_from_source == True delete the rows in the source object.
        if delete_transfered_from_source:
            self._dataset: pd.DataFrame = self._dataset.drop(_temp_df_copy.index)
        
        # Return a fresh MyDataset object.
        return MyDataset(directory_of_data=self.directory_of_data, already_computed_dataframe=_temp_df_copy)
    
    def get_all_distinct_words_in_dataset(self) -> List[str]:
        """Return all the words in each caption of the dataset as a big list of strings (No Repetition).

        Returns:
            (List[str]): All the words in the dataset.
        """
        words = []
        # Iterate over each sample in the dataset.
        for idx,row in self._dataset.iterrows():
            for word in row["comment"]:
                if word not in words:
                    words.append(word)
        return words
    
    def __len__(self) -> int:
        """Evaluate the length of the dataset.
            The length is the number of rows in the dataset.

        Returns:
            int: The legth of the dataset.

        """
        return self._dataset.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[str]]:
        """Get the associated image and caption of a given index.

        Args:
            idx (int): 
                The index associated univocally to a row of the dataset.

        Returns:
            (Tuple[Image.Image, List[str]]): 
                Image and caption of the input index.
        """
        image: Image.Image = Image.open(f"{self.directory_of_data}/{IMAGES_SUBDIRECTORY_NAME}/{self._dataset.iloc[idx]['image_name']}").convert('RGB')
        caption: List[str] = self._dataset.iloc[idx]["comment"]
        
        return image, caption 
    
    # For python > 3.9 -> def pack_minibatch_training(self, data: List[Tuple[Image.Image, List[str]]], vocabulary: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def pack_minibatch_training(self, data: List[Tuple[Image.Image, List[str]]], vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom method for packing a mini-batch for training.

        Args:
            data (List[Tuple[image.Image, List[str]]]): 
                A list of tuples coming from the calls of the __getitem__ method.
                
            vocabulary (Vocabulary): 
                Vocabulary associated to the dataset.

        Returns:
            (Tuple[
                    torch.Tensor,
                    torch.Tensor, 
                    torch.Tensor
                  ]): [`(batch_dim, channels, height, width)`, `(batch_dim,min(MAX_CAPTION_LENGTH,captions[0]))`, `(batch_dim)`]
                  
                Tuple[0]: The images of the mini-batch converted to Tensor.
                Tuple[1]: The caption of each image the mini-batch, the dim 2 depends on the maximum caption length inside the batch. 
                Tuple[2]: The length of each caption +2 for <START> and <END> token.
        """
        # Sort the data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
    
        images, captions = zip(*data)
        
        # Type annotation for zip extraction, no clear way to determine type with this kind of built-in method in a pythonic way.
        images: List[Image.Image] = images
        captions: List[List[str]] = captions
        
        # Trasnform the images from PIL.Image into a pytorch.Tensor
        operations = transforms.Compose([
                transforms.Resize((MyDataset.image_trasformation_parameter["crop"]["size"],MyDataset.image_trasformation_parameter["crop"]["size"])), # Crop a random portion of image and resize it to a given size.
                transforms.RandomHorizontalFlip(p=0.3), # Horizontally flip the given image randomly with a given probability.
                transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.  (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                transforms.Normalize(mean=MyDataset.image_trasformation_parameter["mean"], std=MyDataset.image_trasformation_parameter["std_dev"]),
        ])
        images = list(map(lambda image: operations(image),list(images))) # Out: List[(channels, height, width)]
        # Merge images (from list of 3D tensor to a tensor).
        images = torch.stack(images, 0) #  Out: (batch_dim, channels, height, width)
        
        # Evaluate captions: Devo
        # Q. Why +2?
        # A. For the <START> and <END> Token.
        captions_length = torch.tensor([len(caption)+2 for caption in captions]) # Out: (batch_dim)
        
        # From to words to ids of vocabulary, add <START>.id at beginning and <END>.id at end.
        captions = [vocabulary.translate(caption,"complete") for caption in captions]
        
        # Pad the captions with zeros id == <PAD>.id.
        captions = nn.utils.rnn.pad_sequence(captions, padding_value=0, batch_first=True) # Out: (batch_dim,min(MAX_CAPTION_LENGTH,captions[0]))
        
        
        return images, captions.type(torch.LongTensor), captions_length.type(torch.int32)
    
    # For python > 3.9 -> def pack_minibatch_training(self, data: List[Tuple[Image.Image, List[str]]], vocabulary: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def pack_minibatch_evaluation(self, data: List[Tuple[Image.Image, List[str]]], vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom method for packing a mini-batch for evaluation.

        Args:
            data (List[Tuple[image.Image, List[str]]]): 
                A list of tuples coming from the calls of the __getitem__ method.
                
            vocabulary (Vocabulary): 
                Vocabulary associated to the dataset.

        Returns:
            (Tuple[
                    torch.Tensor,
                    torch.Tensor, 
                    torch.Tensor
                  ]): [`(batch_dim, channels, height, width)`, `(batch_dim,min(MAX_CAPTION_LENGTH,captions[0]))`, `(batch_dim)`]
                  
                Tuple[0]: The images of the mini-batch converted to Tensor.
                Tuple[1]: The caption of each image the mini-batch, the dim 2 depends on the maximum caption length inside the batch. 
                Tuple[2]: The length of each caption +2 for <START> and <END> token.
        """
        # Sort the data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
    
        images, captions = zip(*data)
        
        # Type annotation for zip extraction, no clear way to determine type with this kind of built-in method in a pythonic way.
        images: List[Image.Image] = images
        captions: List[List[str]] = captions
        
        # Trasnform the images from PIL.Image into a pytorch.Tensor)
        
        operations = transforms.Compose([
                transforms.Resize((MyDataset.image_trasformation_parameter["crop"]["size"], MyDataset.image_trasformation_parameter["crop"]["size"])),  # Crops the given image at the center.
                transforms.ToTensor(),
                transforms.Normalize(mean=MyDataset.image_trasformation_parameter["mean"], std=MyDataset.image_trasformation_parameter["std_dev"])
        ])

        images = list(map(lambda image: operations(image),list(images))) # Out: List[(channels, height, width)]
        # Merge images (from list of 3D tensor to a tensor).
        images = torch.stack(images, 0) #  Out: (batch_dim, channels, height, width)
        
        # Evaluate captions: Devo
        # Q. Why +2?
        # A. For the <START> and <END> Token.
        captions_length = torch.tensor([len(caption)+2 for caption in captions]) # Out: (batch_dim)
        
        # From to words to ids of vocabulary, add <START>.id at beginning and <END>.id at end.
        captions = [vocabulary.translate(caption,"complete") for caption in captions]
        
        # Pad the captions with zeros id == <PAD>.id.
        captions = nn.utils.rnn.pad_sequence(captions, padding_value=0, batch_first=True) # Out: (batch_dim,min(MAX_CAPTION_LENGTH,captions[0]))
        
        
        return images, captions.type(torch.LongTensor), captions_length.type(torch.int32)
        