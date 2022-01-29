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

        Args (Suggested):
            projection_size (int): The dimension of projection into the space of RNN (Could be the input or the hidden state).
            
            device (str, optional): The device on which the operations will be performed. Default "cpu".
        """
        super(IEncoder, self).__init__()
        
    def forward(self, *args) -> torch.Tensor:
        """Interface of forward operation of the nn

        Args (Suggestes):
            images (torch.tensor): The tensor of the image in the form (Batch Size, Channels, Width, Height)

        Returns:
            [torch.tensor]: Features Projection in the form (Batch Size, Projection Dim.)
        """
        pass