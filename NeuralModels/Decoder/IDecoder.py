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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List

class IDecoder(nn.Module):
    """
        Class interface for a LSTM unit 
        Args are intended as suggested.
    """
    
    def __init__(self, *args):
        """Define the interface of a generic constructor for the Decoder Net.

        Args (Suggested):
        
            hidden_dim (int): 
                The Capacity of the LSTM Cell. 
                
            padding_index (int): 
                The index of the padding id, given from the vocabulary associated to the dataset.
                
            vocab_size (int)): 
                The size of the vocabulary associated to the dataset.
                
            embedding_dim (int): 
                The number of features associated to a word.
                
            device (str, optional): Default "cpu"
                The device on which the operations will be performed. 
        """
        super(IDecoder, self).__init__()                

    def forward(self, *args) -> Tuple[torch.Tensor, List[int]]:
        """Interface for the forward operation of the RNN.
                  
        Args (Suggested): 
        
            images (torch.Tensor): `(batch_dim, encoder_dim)`
                The features associated to each image of the batch. 
            
            captions (torch.Tensor): `(batch_dim, max_captions_length, embedding_dim)`
                The caption associated to each image of the batch. 
                    _REMARK Each caption is in the full form: <START> + .... + <END>_
                
            caption_length (list(int)): 
                The length of each caption in the batch.
            
        Returns:    `[(batch_size, max_captions_length, vocab_size), list(int)]`
        
            (torch.Tensor): The hidden state of each time step from t_1 to t_N. 
            
            (list(int)): The length of each decoded caption. 
                REMARK The <START> is provided as input at t_0.
                REMARK The <END> token will be removed from the input of the LSTM.
        """             
        pass
    
    def generate_caption(self, *args) -> torch.Tensor:
        """ Interface for generate a caption

        Args (Suggested):
        
            images (torch.Tensor): `(1, encoder_dim)`
                The features associated to the image. 
                
            max_caption_length (int): 
                The maximum ammisible length of the caption.

        Returns:
        
            (torch.Tensor): `(1, <variable>)`
                The caption associated to the image given. 
                    REMARK It includes <START> at t_0 by default.
        """
        pass