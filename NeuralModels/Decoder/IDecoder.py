#####   INTERFACE CLASS DON'T USE IT (You at most use only as Type Hint), JUST READ IT.
################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List

class IDecoder(nn.Module):
    """
        Class interface for LSTM unit 
        Args are intended as suggested.
    """
    
    def __init__(self, *args):
        """Define the interface of a generic constructor for the Decoder Net

        Args (Suggested):
            hidden_size (int): The Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_size (int): The number of dimension associated to the input of the LSTM cell
            device (str, optional): The device on which the operations will be performed. Default "cpu"
        """
        super(IDecoder, self).__init__()                

    def forward(self, *args) -> Tuple[torch.tensor, List[int]]:
        """Interface for the forward operation of the RNN.
                input of the LSTM cell for each time step:
                    t_{-1}: feature vector 
                    t_0: Deterministict <SOS> 
                    .
                    .
                    .
                    t_{N-1}: The embedding vector associated to the S_{N-1} id.  
                    
        Args (Suggested):
            features (torch.tensor): The features associated to each element of the batch. (batch_size, embed_size)
            
            captions (torch.tensor): The caption associated to each element of the batch. (batch_size, max_captions_length, word_embedding)
                REMARK Each caption is in the full form: <SOS> + .... + <EOS>
                
            caption_length ([int]): The length of each caption in the batch.
            
        Returns:
            (torch.tensor): The hidden state of each time step from t_1 to t_N. (batch_size, max_captions_length, vocab_size)
            
            (list(int)): The length of each decoded caption. 
                REMARK The <SOS> is provided as input at t_0.
                REMARK The <EOS> token will be removed from the input of the LSTM.
        """             
        pass
    
    def generate_caption(self, *args) -> torch.tensor:
        """ Interface for generate a caption

        Args (Suggested):
            feature (torch.tensor): The features vector (1, embedding_size)
            captions_length (int): The length of the caption

        Returns:
            torch.tensor: The caption associated to the image given. 
                    It includes <SOS> at t_0 by default.
        """
        pass