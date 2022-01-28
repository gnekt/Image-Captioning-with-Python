#####################################################
## DISCLAIMER: IL CODICE E` ESSENZIALMENTE HARDCODED, SOLO DI TESTING, NON RISPETTA I CANONI DELLO SGD, CERCO DI CAPIRE SOLO SE FUNZIONA! 
# NON GIUDICARLO GENTILMENTE, SO CHE NON VA` FATTO COSI`, POI LO SISTEMO :)
##
##
##  pip install torch==1.3.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from typing import Tuple,List
from Dataset import MyDataset
from Vocabulary import Vocabulary

class EncoderCNN(nn.Module):
    def __init__(self, projection_size: int, device: str = "cpu"):
        """Constructor of the Encoder NN

        Args:
            projection_size (int): The dimension of projection into the space of RNN (Could be the input or the hidden state).
            
            device (str, optional): The device on which the operations will be performed. Default "cpu".
        """
        super(EncoderCNN, self).__init__()
        
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # Freezing weights 
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        
        self.linear = nn.Linear(resnet.fc.in_features, projection_size) # define a last layer 
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward operation of the nn

        Args:
            images (torch.tensor): The tensor of the image in the form (Batch Size, Channels, Width, Height)

        Returns:
            [torch.tensor]: Features Projection in the form (Batch Size, Projection Dim.)
        """
        # To Do Add dimensionality 
        features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1).to(self.device)
        features = self.linear(features)
        
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, padding_index: int, vocab_size: int, embedding_size: int, device: str = "cpu"):
        """Define the constructor for the RNN Net

        Args:
            hidden_size (int): The Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_size (int): The number of dimension associated to the input of the LSTM cell
            device (str, optional): The device on which the operations will be performed. Default "cpu"
        """
        super(DecoderRNN, self).__init__()
    
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
                    t_{-1}: feature vector 
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
            (torch.tensor): The hidden state of each time step from t_1 to t_N. (batch_size, max_captions_length, vocab_size)
            
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

    
class CaRNetvHC(nn.Module):
    
    def __init__(self, hidden_size: int, padding_index: int, vocab_size: int, embedding_size: int, device: str = "cpu"):
        """Create the CaRNet 

        Args:
            hidden_size (int): The Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_size (int): The number of dimension associated to the input of the LSTM cell
            device (str, optional): The device on which the net does the computation. Defaults to "cpu".
        """
        
        super(CaRNetvHC, self).__init__()
        self.padding_index = padding_index
        self.device = torch.device(device)
        
        # Define Encoder and Decoder
        self.C = EncoderCNN(hidden_size, device)
        self.R = DecoderRNN(hidden_size, padding_index, vocab_size, embedding_size, device)

        self.C.to(self.device)
        self.R.to(self.device)
        
    def save(self, file_path: str) -> bool:
        """Save the net in non-volatile memory

        Args:
            file_name (str): Relative path to save the net. Ex. "home/pippo/saved"

        Returns:
            bool: If True: Net saved correctly. False otherwise.
        """
        try:
            torch.save(self.C.state_dict(), f"{file_path}/CaRNetvHC_C.pth")
            torch.save(self.R.state_dict(), f"{file_path}/CaRNetvHC_R.pth")
        except Exception as ex:
            print(ex)
            return False
        return True

    def load(self, file_path: str) -> bool:
        """Load the net from non-volatile memory into RAM

        Args:
            file_name (str): Relative path of the net. Ex. "home/pippo/saved"

        Returns:
            bool: If True: Net loaded correctly. False otherwise.
        """
        
        # since our classifier is a nn.Module, we can load it using pytorch facilities (mapping it to the right device)
        self.C.load_state_dict(torch.load(f"{file_path}/CaRNetvHC_C.pth", map_location=self.device))
        self.R.load_state_dict(torch.load(f"{file_path}/CaRNetvHC_R.pth", map_location=self.device))
    
    def forward(self, images: torch.tensor, captions: torch.tensor) -> torch.tensor:
        """Provide images to the net for retrieve captions

        Args:
            images (torch.tensor): The images of the batch. (Batch Size, Channels, Width, Height)
            captions (torch.tensor): (Batch Size, Max_Captions_Length). 
                ASSUMPION: The captions are padded with <PAD> Token

        Returns:
            (torch.tensor): The hidden state of each time step from t_1 to t_N. (batch_size, max_captions_length, vocab_size)
        """
        features = self.C(images)
        return self.R(features, captions)

    def __accuracy(self, outputs: torch.tensor, labels: torch.tensor, captions_length: List[int]) -> float:
        """Evaluate the accuracy of the Net.
                Assumption: outputs and labels have same shape and already padded.

        Args:
            outputs (torch.tensor): [description]
            labels (torch.tensor): [description]
            captions_length (list): [description]

        Returns:
            float: The accuracy of the Net
        """
        
        # We could subtract labels.ids to outputs.ids tensor, all the values different from 0 (output_caption_id != target_caption_id) are mismatch!
        
        # computing the accuracy 
        
        # To Do add dimensionality 
        outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, captions_length.cpu(), batch_first=True).to(self.device)
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, captions_length.cpu(), batch_first=True).to(self.device)
        right_predictions =  outputs.data - labels.data == 0
        
        acc = right_predictions.to(torch.float32).sum(axis=0) / right_predictions.shape[0]  
        return acc
    
        # TO DO: Devo usare la confusion matrix????????? 
    
    def train(self, train_set: MyDataset, validation_set: MyDataset, lr: float, epochs: int, vocabulary: Vocabulary):
        """[summary]

        Args:
            train_set (MyDataset): [description]
            validation_set (MyDataset): [description]
            lr (float): [description]
            epochs (int): [description]
            vocabulary (Vocabulary): [description]
        """
        
        # Initialize Loss: CrossEntropyLoss -> Softmax + NegativeLogLikelihoodLoss 
        # Q. Why ignore_index is setted to <SOS> instead of <PAD>?
        # A. In the training, both output of the CaRNet and Target label start as padded tensor, but when we compute the loss it will evaluate the tensor with pack_padded_sequence.
        #       And since <SOS> token is hardcoded as output at t_0 we could avoid the computation of loss on it, since will be 0 fover.                     
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<SOS>"],reduction="sum").cuda() if self.device.type == "cuda"  \
                                            else nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<SOS>"],reduction="sum")
        
        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data
        best_epoch = -1  # the epoch in which the best accuracy above was computed
        
        # ensuring the classifier is in 'train' mode (pytorch)
        self.C.train()
        self.R.train()

        # creating the optimizer
        optimizer = torch.optim.Adam(list(self.R.parameters()) + list(self.C.parameters()), lr)

        # loop on epochs!
        for e in range(0, epochs):

            # epoch-level stats (computed by accumulating mini-batch stats)
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0

            for images,captions_ids,captions_length in  train_set:
                optimizer.zero_grad() 
                
                batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size') -> last batch truncated
                epoch_num_train_examples += batch_num_train_examples
                
                
                images = images.to(self.device)
                captions_ids = captions_ids.to(self.device) # captions > (B, L)
                captions_length = captions_length.to(self.device)
                
                # computing the network output on the current mini-batch
                features = self.C(images)
                outputs, outputs_length = self.R(features, captions_ids, captions_length) # outputs > (B, L, |V|); 
                
                outputs = pack_padded_sequence(outputs, captions_length.cpu(), batch_first=True)  #(Batch, MaxCaptionLength, |Vocabulary|) -> (Batch * CaptionLength, |Vocabulary|)
                
                targets = pack_padded_sequence(captions_ids, captions_length.cpu(), batch_first=True) #(Batch, MaxCaptionLength) -> (Batch * CaptionLength)
                
                
                loss = criterion(outputs.data, targets.data)
                
                # computing gradients and updating the network weights
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights
                
                with torch.no_grad():
                    self.C.eval()
                    self.R.eval()
                    features = self.C(images)
                    import random
                    numb = random.randint(0,2)
                    caption = self.R.generate_caption(features[numb],30)
                    print(vocabulary.rev_translate(captions_ids[numb]))
                    print(vocabulary.rev_translate(caption[0]))
                    self.C.train()
                    self.R.train()
                
                with torch.no_grad():
                    self.C.eval()
                    self.R.eval()
                    
                    # Compute captions as ids for all the training images
                    projections = self.C(images)
                    
                    captions_output = torch.zeros((projections.shape[0],captions_ids.shape[1])).to(self.device)
                    
                    for idx,projection in enumerate(range(projections.shape[0])):
                        _caption_no_pad = self.R.generate_caption(projections[idx],captions_ids.shape[1])
                        captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                        # Fill the remaining portion of caption eventually with zeros
                        # Accuracy is not altered since if the length of caption is smaller than the captions_target_ids(padded), feed it with PAD is valid.

                    captions_output_padded = captions_output.type(torch.int32).to(self.device) # From list of tensors to tensors
                    
                    # computing performance
                    batch_train_acc = self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length)

                    # accumulating performance measures to get a final estimate on the whole training set
                    epoch_train_acc += batch_train_acc * batch_num_train_examples

                    # accumulating other stats
                    epoch_train_loss += loss.item() * batch_num_train_examples
                    self.C.train()
                    self.R.train()
                    
                    # printing (mini-batch related) stats on screen
                    print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_acc))
                    
            val_acc = self.eval_classifier(validation_set)

            # # saving the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                
                self.save("/content/drive/MyDrive/Progetti/Neural Networks/.saved")

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                + (", BEST!" if best_epoch == e + 1 else ""))
                .format(e + 1, epochs, epoch_train_loss,
                        epoch_train_acc / epoch_num_train_examples, val_acc))


    def eval_classifier(self, data_set):
        """Evaluate the classifier on the given data set."""

        # checking if the classifier is in 'eval' or 'train' mode (in the latter case, we have to switch state)
        training_mode_originally_on = self.C.training and self.R.training
        if training_mode_originally_on:
            self.C.eval()
            self.R.eval()  # enforcing evaluation mode

        

        with torch.no_grad():  # keeping off the autograd engine

            # loop on mini-batches to accumulate the network outputs (creating a new iterator)
            for images,captions_ids,captions_length  in data_set:
                images = images.to(self.device)
                
                captions_ids = captions_ids.to(self.device)

                projections = self.C(images)
                        
                captions_output = torch.zeros((projections.shape[0],captions_ids.shape[1])).to(self.device)
                
                for idx,projection in enumerate(range(projections.shape[0])):
                    _caption_no_pad = self.R.generate_caption(projections[idx],captions_ids.shape[1])
                    captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                    # Fill the remaining portion of caption eventually with zeros
                    # Accuracy is not altered since if the length of caption is smaller than the captions_target_ids(padded), feed it with PAD is valid.

                captions_output_padded = captions_output.type(torch.int32).to(self.device) # From list of tensors to tensors
                
                # computing performance
                acc = self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length)

        if training_mode_originally_on:
            self.C.train()  # restoring the training state, if needed
            self.R.train()
        return acc
    
# Example of usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset", percentage=1)
    v = Vocabulary(ds,reload=True) 
    dc = ds.get_fraction_of_dataset(percentage=70, delete_transfered_from_source=True)
    df = ds.get_fraction_of_dataset(percentage=30, delete_transfered_from_source=True)
    # use dataloader facilities which requires a preprocessed dataset
       
    
    dataloader_training = DataLoader(dc, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: ds.pack_minibatch_training(data,v))
    
    dataloader_evaluation = DataLoader(df, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: ds.pack_minibatch_evaluation(data,v))
    
    net = CaRNetvHC(512,0,len(v.word2id.keys()),v.embeddings.shape[1],"cuda:0")
    #net.load("CaRNetvI")
    net.train(dataloader_training,dataloader_evaluation,1e-3,500,v)
