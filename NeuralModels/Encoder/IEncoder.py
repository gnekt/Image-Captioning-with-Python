# MIT License

# Copyright (c) 2022 christiandimaio

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



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