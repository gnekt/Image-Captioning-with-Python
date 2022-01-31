#####   INTERFACE CLASS DON'T USE IT (You at most use only as Type Hint), JUST READ IT.
################################################################

import torch.nn as nn
import torch
import torchvision.models as models

class IEncoder(nn.Module):
    """
        Interface for a generic Encoder
    """
    def __init__(self,  *args):
        """Constructor of the Encoder NN

        Args:
            encoder_dim (int): 
                The dimensionality of the features vector extracted from the image
            
            device (str, optional): Default "cpu".
                The device on which the operations will be performed.
        """
        super(IEncoder, self).__init__()
        
    def forward(self, *args) -> torch.Tensor:
        """Interface of forward operation of the nn

        Args:
            images (torch.tensor):  `(batch_dim, channels, heigth, width)`
                The tensor of the images.

        Returns:
            [torch.tensor]: `(batch_dim, encoder_dim)`
                Features Projection for each image in the batch.
        """
        pass