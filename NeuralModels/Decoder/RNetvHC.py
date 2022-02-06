import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List

class RNetvHC(nn.Module):
    """
        Class implementing LSTM unit with Cell and Hidden state initialized with custom features vector 
    """
    def __init__(self, hidden_dim: int, padding_index: int, vocab_size: int, embedding_dim: int, device: str = "cpu"):
        """Define the constructor for the RNN Net

        Args:
        
            hidden_dim (int): 
                Capacity of the LSTM Cell.
                
            padding_index (int): 
                The index of the padding id, given from the vocabulary associated to the dataset.
                
            vocab_size (int): 
                The size of the vocabulary associated to the dataset.
                
            embedding_dim (int): 
                The number of features associated to a word.
                
            device (str, optional): Default "cpu"
                The device on which the operations will be performed. 
        """
        super(RNetvHC, self).__init__()

        print(f"Construction of RNetvHC:\n \
                LSTM Capacity: {hidden_dim},\n \
                Padding Index: {padding_index},\n \
                Vocabulary Size: {vocab_size},\n \
                Embedding dimension: {embedding_dim},\n \
                Device: {device}")
        
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # Embedding layer that turns words into a vector.
        self.words_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        
        # The linear layer that maps the hidden state
        # to the number of words we want as output = vocab_size
        self.linear_1 = nn.Linear(hidden_dim, vocab_size)
                

    def forward(self, images: torch.Tensor, captions: torch.Tensor, captions_length: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """Compute the forward operation of the RNN.
                input of the LSTM cell for each time step:
                    t_{-1}: NONE 
                    t_0: Deterministict <START> 
                    .
                    .
                    .
                    t_{N-1}: The embedding vector associated to the S_{N-1} id.

        Args (Suggested): 
        
            images (torch.Tensor): `(batch_dim, encoder_dim)`
                The features associated to each image of the batch. 
            
            captions (torch.Tensor): `(batch_dim, max_captions_length, embedding_dim)`
                The caption associated to each image of the batch. 
                    _REMARK Each caption is in the full form: <START> + .... + <END>_
                    REMARK The Tensor is padded with zeros
                    
            caption_length (List(int)): 
                The length of each caption in the batch.
            
        Returns:    `[(batch_dim, max_captions_length, vocab_size), List(int)]`
        
            (torch.Tensor): 
                The output of LSTM for each time step from t_1 to t_N, + <START> at t_0
                    REMARK <START> is the 1st element in the output caption for each element in batch.
                    
            (List(int)): The length of each decoded caption. 
                REMARK The <START> is provided as input at t_0.
                REMARK The <END> token will be removed from the input of the LSTM.
        """             
        # Check if encoder_dim and self.hidden_dim are equal, assert by construction
        if images.shape[1] != self.hidden_dim:
            raise ValueError("The dimensionality of the encoder output is not equal to the dimensionality of the hidden state.")
        
        # Retrieve batch size 
        batch_dim = images.shape[0] 
        
        # Create embedded word vector for each word in the captions
        inputs = self.words_embedding(captions) # In: (batch_dim, max_captions_length, embedding_dim) ->  Out: (batch_dim, captions length, embedding_dim)
        
        # Initialize the hidden state and the cell state at time t_{-1}
        _h, _c = (images, images) #  In: ((batch_dim, hidden_dim),(batch_dim, hidden_dim)) -> Out ((batch_dim, hidden_dim), (batch_dim, hidden_dim))
        
        # Deterministict <START> Output as first word of the caption t_{0}
        start = torch.zeros(self.vocab_size)
        start[1] = 1
        start = start.to(self.device)  # Out: (1, vocab_size)
        
        # Bulk insert of <START> to all the elements of the batch 
        outputs = start.repeat(batch_dim,1,1).to(self.device) # Out: (batch_dim, 1, vocab_size)
          
        # Feed LSTMCell with image features and retrieve the state
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
                
        for idx in range(0,inputs.shape[1]): 
            _h, _c = self.lstm_unit(inputs[:,idx,:], (_h,_c))  # inputs[:,idx,:]: for all the captions in the batch, pick the embedding vector of the idx-th word in all the captions
            _outputs = self.linear_1(_h) # In: (batch_dim, hidden_dim), Out: (batch_dim, vocab_size)
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1) # Append in dim `1` the output of the LSTMCell
        
        return outputs, list(map(lambda length: length-1, captions_length))  
    
    def generate_caption(self, image: torch.Tensor, captions_length: int) -> torch.Tensor:
        """Given the features vector of the image, perform a decoding (Generate a caption)

        Args:
        
            image (torch.Tensor): `(1, encoder_dim)`
                The features associated to the image. 
                
            max_caption_length (int): 
                The maximum ammisible length of the caption.

        Returns:
        
            (torch.Tensor): `(1, <variable>)`
                The caption associated to the image given. 
                    REMARK It includes <START> at t_0 by default.
        """
        
        sampled_ids = [torch.tensor([1]).to(self.device)] # Hardcoded <START>
        input = self.words_embedding(torch.LongTensor([1]).to(torch.device(self.device))).reshape((1,-1)) # Out: (1, embedding_dim)
        with torch.no_grad(): 
            _h ,_c = (image,image)
            for _ in range(captions_length-1):
                _h, _c = self.lstm_unit(input, (_h ,_c))           # Out : ((1, hidden_dim) , (1, hidden_dim))
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if self.device.type == "cuda" else   F.softmax(outputs,dim=1).max(1)  # predicted: The predicted id
                sampled_ids.append(predicted)
                input = self.words_embedding(predicted)                       # Out: (1, embedding_dim)
                input = input.to(torch.device(self.device))                     # Out: (1, embedding_dim)
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (1, captions_length)
        return sampled_ids