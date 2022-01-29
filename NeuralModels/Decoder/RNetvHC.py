import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List

class RNetvHC(nn.Module):
    """
        Class implementing LSTM unit with Cell and Hidden state initialized with custom features vector 
    """
    def __init__(self, hidden_size: int, padding_index: int, vocab_size: int, embedding_size: int, device: str = "cpu"):
        """Define the constructor for the RNN Net

        Args:
            hidden_size (int): The Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_size (int): The number of dimension associated to the input of the LSTM cell
            device (str, optional): The device on which the operations will be performed. Default "cpu"
        """
        super(RNetvHC, self).__init__()

        print(f"Construction of RNetvHC:\n\t Hidden Number of Dimensions: {hidden_size},\n\t Padding Index: {padding_index},\n\t Vocabulary Size: {vocab_size},\n\t Embedding Number of Dimension: {embedding_size},\n\t Device: {device}")
        
        self.device = torch.device(device)
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_size, hidden_size)
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_size, vocab_size)
                

    def forward(self, features: torch.tensor, captions: torch.tensor, captions_length: List[int]) -> Tuple[torch.tensor, List[int]]:
        """Compute the forward operation of the RNN.
                input of the LSTM cell for each time step:
                    t_{-1}: NONE 
                    t_0: Deterministict <SOS> 
                    .
                    .
                    .
                    t_{N-1}: The embedding vector associated to the S_{N-1} id.

        Args:
            features (torch.tensor): The features associated to each element of the batch. (batch_size, embed_size)
            
            captions (torch.tensor): The caption associated to each element of the batch. (batch_size, max_captions_length, word_embedding)
                REMARK Each caption is in the full form: <SOS> + .... + <EOS>
                
            caption_length ([int]): The length of each caption in the batch.    
        Returns:
            (torch.tensor): The hidden state of each time step from t_1 to t_{MaxN}. (batch_size, max_captions_length, vocab_size)
            
            (list(int)): The length of each decoded caption. 
                REMARK The <SOS> is provided as input at t_0.
                REMARK The <EOS> token will be removed from the input of the LSTM.
        """             
        
        # Retrieve batch size 
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        
        # Create embedded word vector for each word in the captions
        inputs = self.word_embeddings(captions) # In:       Out: (batch_size, captions length, embed_size)
        
        # Initialize the hidden state and the cell state at time t_{-1}
        _h, _c = (features, features) # _h : (Batch size, Hidden size), _c : (Batch size, Hidden size)
        
        # Deterministict <SOS> Output as first word of the caption t_{0}
        start = self.word_embeddings(torch.LongTensor([1]).to(self.device))  # Get the embeddings of the token <SOS>
        
        # Bulk insert of <SOS> embeddings to all the elements of the batch 
        outputs = start.repeat(batch_size,1,1).to(self.device) 
          
        # Feed LSTMCell with image features and retrieve the state
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
        
        # Since the sequences are padded, how the forward is performed? Since the <EOS> don't need to be feeded as input?
        # The assumption is that the decode captions will have a length 
        
        for idx in range(0,inputs.shape[1]): 
            _h, _c = self.lstm_unit(inputs[:,idx,:], (_h,_c))  # inputs[:,idx,:]: for all the captions in the batch, pick the embedding vector of the idx-th word in all the captions
            _outputs = self.linear_1(_h) 
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1) # Append in dim `1` the output of the LSTMCell for all the elements in batch
        
        return outputs, list(map(lambda length: length-1, captions_length))  
    
    def generate_caption(self, feature: torch.tensor, captions_length: int) -> torch.tensor:
        """Given the features vector retrieved by the encoder, perform a decoding (Generate a caption)

        Args:
            feature (torch.tensor): The features vector (1, embedding_size)
            captions_length (int): The length of the caption

        Returns:
            torch.tensor: The caption associated to the image given. 
                    It includes <SOS> at t_0 by default.
        """
        
        sampled_ids = [torch.tensor([1]).to(self.device)] # Hardcoded <SOS>
        input = self.word_embeddings(torch.LongTensor([1]).to(torch.device(self.device))).reshape((1,-1))
        with torch.no_grad(): 
            _h ,_c = (feature.unsqueeze(0),feature.unsqueeze(0))
            for _ in range(captions_length-1):
                _h, _c = self.lstm_unit(input, (_h ,_c))           # _h: (1, 1, hidden_size)
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if self.device.type == "cuda" else   F.softmax(outputs,dim=1).max(1)  # predicted: The predicted id
                sampled_ids.append(predicted)
                input = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
                input = input.to(torch.device(self.device))                       # inputs: (batch_size, 1, embed_size)
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (1, captions_length)
        return sampled_ids