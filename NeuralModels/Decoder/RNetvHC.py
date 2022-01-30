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
            hidden_size (int): Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_size (int): Size associated to the input of the LSTM cell
            device (str, optional): The device on which the operations will be performed. Default "cpu"
        """
        super(RNetvHC, self).__init__()

        print(f"Construction of RNetvHC:\n\t Hidden Number of Dimensions: {hidden_dim},\n\t Padding Index: {padding_index},\n\t Vocabulary Size: {vocab_size},\n\t Embedding Number of Dimension: {embedding_dim},\n\t Device: {device}")
        
        self.device = torch.device(device)
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_dim, vocab_size)
                

    def forward(self, images: torch.tensor, captions: torch.tensor, captions_length: List[int]) -> Tuple[torch.tensor, List[int]]:
        """Compute the forward operation of the RNN.
                input of the LSTM cell for each time step:
                    t_{-1}: NONE 
                    t_0: Deterministict <SOS> 
                    .
                    .
                    .
                    t_{N-1}: The embedding vector associated to the S_{N-1} id.

        Args:
            images (torch.tensor): The features vector associated to each image of the batch. (batch_size, embeddings_dim)
            
            captions (torch.tensor): The caption associated to each element of the batch. (batch_size, max_captions_length, word_embedding)
                REMARK Each caption is in the full form: <SOS> + .... + <EOS>
                
            caption_length (List[int]): The length of each caption in the batch.    
        Returns:
            (torch.tensor): The hidden state of each time step from t_1 to t_{MaxN}. (batch_size, max_captions_length, vocab_size)
            
            (List[int]): The length of each decoded caption. 
                REMARK The <SOS> is provided as input at t_0.
                REMARK The <EOS> token will be removed from the input of the LSTM.
        """             
        # Check if encoder_dim and self.hidden_dim are equal, assert by construction
        if images.shape[1] != self.hidden_dim:
            raise ValueError("The dimensionality of the encoder output is not equal to the dimensionality of the hidden state.")
        
        # Retrieve batch size 
        batch_size = images.shape[0] 
        
        # Create embedded word vector for each word in the captions
        inputs = self.word_embeddings(captions) # In: (batch_size, max_captions_length, embeddings_dim) ->  Out: (batch_size, captions length, embeddings_dim)
        
        # Initialize the hidden state and the cell state at time t_{-1}
        _h, _c = (images, images) #  In: ((batch_size, hidden_dim),(batch_size, hidden_dim)) -> Out ((batch_size, hidden_dim), (batch_size, hidden_dim))
        
        # Deterministict <SOS> Output as first word of the caption t_{0}
        start = self.word_embeddings(torch.LongTensor([1]).to(self.device))  # Out: (1, embeddings_dim)
        
        # Bulk insert of <SOS> embeddings to all the elements of the batch 
        outputs = start.repeat(batch_size,1,1).to(self.device)  # Out: (batch_size, 1, embeddings_dim)
          
        # Feed LSTMCell with image features and retrieve the state
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
        
        # Since the sequences are padded, how the forward is performed? Since the <EOS> don't need to be feeded as input?
        # The assumption is that the decode captions will have a length 
        
        for idx in range(0,inputs.shape[1]): 
            _h, _c = self.lstm_unit(inputs[:,idx,:], (_h,_c))  # inputs[:,idx,:]: for all the captions in the batch, pick the embedding vector of the idx-th word in all the captions
            _outputs = self.linear_1(_h) # In: (batch_size, hidden_dim)
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1) # Append in dim `1` the output of the LSTMCell
        
        return outputs, list(map(lambda length: length-1, captions_length))  
    
    def generate_caption(self, image: torch.tensor, captions_length: int) -> torch.tensor:
        """Given the features vector retrieved by the encoder, perform a decoding (Generate a caption)

        Args:
            image (torch.tensor): The features vector (1, embeddings_dim)
            captions_length (int): The length of the caption

        Returns:
            torch.tensor: The caption associated to the image given. 
                    It includes <SOS> at t_0 by default.
        """
        
        sampled_ids = [torch.tensor([1]).to(self.device)] # Hardcoded <SOS>
        input = self.word_embeddings(torch.LongTensor([1]).to(torch.device(self.device))).reshape((1,-1)) # Out: (batch_size, embeddings_dim)
        with torch.no_grad(): 
            _h ,_c = (image.unsqueeze(0),image.unsqueeze(0))
            for _ in range(captions_length-1):
                _h, _c = self.lstm_unit(input, (_h ,_c))           # Out : ((1, 1, hidden_size) , (1, 1, hidden_size))
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if self.device.type == "cuda" else   F.softmax(outputs,dim=1).max(1)  # predicted: The predicted id
                sampled_ids.append(predicted)
                input = self.word_embeddings(predicted)                       # Out: (batch_size, embeddings_dim)
                input = input.to(torch.device(self.device))                      
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (1, captions_length)
        return sampled_ids