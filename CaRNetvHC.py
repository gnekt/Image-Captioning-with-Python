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

device="cpu"
class EncoderCNN(nn.Module):
    def __init__(self, projection_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, projection_size) 
        
    def forward(self, images):
        features = self.resnet(images) 
        features = features.reshape(features.size(0), -1) # (Batch Size, Embedding Dim.)
        features = self.linear(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, padding_index, vocab_size, embedding_size):
        """[summary]

        Args:
            hidden_size ([type]): [description]
            padding_index ([type]): [description]
            vocab_size ([type]): [description]
            embedding_size ([type]): [description]
        """
        super(DecoderRNN, self).__init__()
    
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_size, hidden_size)
        
        self.hidden_size = hidden_size
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_size, vocab_size)
                

    def forward(self, features, captions):
        """[summary]

        Args:
            features (torch.tensor(batch_size, hidden_size)): [description]
            captions (torch.tensor(batch_size, max_captions_length, word_embedding)): [description]

        Returns:
            [torch.tensor(batch_size, max_captions_length, vocab_size)]: [description]
        """             
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        
        # Create embedded word vector for each word in the captions
        inputs = self.word_embeddings(captions) # In:       Out: (batch_size, captions length, embed_size)
        
        # Feed LSTMCell with image features and retrieve the state
        
        _h, _c = tuple( features, features) # _h : (Batch size, Hidden size)
        
        # Deterministict <SOS> Output as first word of the caption :)
        start = torch.zeros(self.word_embeddings.num_embeddings)
        start[1] = 1
        outputs = start.repeat(batch_size,1,1).to(torch.device(device)) # Bulk insert of <SOS> embeddings to all the elements of the batch 
          
        
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
        
        # Since the sequences are padded, how the forward is performed? Since the <EOS> don't need to be feeded as input?
        # The assumption is that the captions are of lenght N-1, so the captions provide by external as input are without <EOS> token
        
        for idx in range(0,inputs.shape[1]): 
            _h, _c = self.lstm_unit(inputs[:,idx,:], (_h,_c))
            _outputs = self.linear_1(_h)
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1)
        
        return outputs # (Batch Size, N, |Vocabulary|)
    
    def generate_caption(self, features, max_caption_length):
        """Generate captions for given image features using greedy search."""
       
        sampled_ids = [torch.tensor([1]).to(torch.device(device))] # Hardcoded <SOS>
        input = self.word_embeddings(torch.LongTensor([1]).to(torch.device(device))).reshape((1,-1))
        with torch.no_grad(): 
            _h, _c = tuple( features.unsqueeze(0), features.unsqueeze(0))
            for _ in range(max_caption_length-1):
                _h, _c = self.lstm_unit(input, (_h ,_c))           # _h: (1, 1, hidden_size)
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if device == "cuda" else   F.softmax(outputs,dim=1).max(1)                # predicted: (batch_size)
                sampled_ids.append(predicted)
                input = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
                input = input.to(torch.device(device))                       # inputs: (batch_size, 1, embed_size)
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    
class CaRNet1(nn.Module):
    
    def __init__(self, hidden_size, padding_index, vocab_size, embedding_size, device = "cpu"):
        """[summary]

        Args:
            hidden_size ([type]): [description]
            padding_index ([type]): [description]
            vocab_size ([type]): [description]
            embedding_size ([type]): [description]
        """
        super(CaRNet1, self).__init__()
        self.padding_index = padding_index
        self.device = torch.device(device)
        self.C = EncoderCNN(embedding_size)
        self.R = DecoderRNN(hidden_size, padding_index, vocab_size, embedding_size)

        self.C.to(self.device)
        self.R.to(self.device)
        
    def save(self, file_name):
        """Save the classifier."""
        torch.save(self.C.state_dict(), f".saved/vHC/{file_name}_C.pth")
        torch.save(self.R.state_dict(), f".saved/vHC/{file_name}_R.pth")

    def load(self, file_name):
        """Load the classifier."""

        # since our classifier is a nn.Module, we can load it using pytorch facilities (mapping it to the right device)
        self.C.load_state_dict(torch.load(f".saved/vHC/{file_name}_C.pth", map_location=self.device))
        self.R.load_state_dict(torch.load(f".saved/vHC/{file_name}_R.pth", map_location=self.device))
    
    def forward(self,images,captions):
        features = self.C(images)
        return self.R(features, captions)
    
    def __accuracy(self, outputs, labels):
        """[summary]

        Args:
            outputs ([type]): [description]
            labels ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Assume outputs and labels have same shape and already padded
        # We could subtract labels.ids to outputs.ids tensor, all the values different from 0 (output_caption_id != target_caption_id) are mismatch!
        # With this technique we evaluate all major case:
        # 1) Output caption is longer than expected : Output.ID - <PAD>.ID != 0
        # 2) Output is less longer than expect : <PAD>.ID - Target.ID != 0
        # 3) Output has equal dimension but different label : Output.ID - Target.ID != 0, 
        # Hp. 1 : Output<PAD>.ID - Target<PAD>.ID = 0 need to be considered as good match because it means that both output and target end before this token
        # Hp. 2 : Both Outputs and Target need to be dropped on the first word because <SOS> is evaluated in a deterministic fashion :)
        # computing the accuracy
        
        right_predictions = torch.eq(outputs[:,1:], labels[:,1:])
        acc = torch.mean(right_predictions.to(torch.float32).sum(axis=1) / right_predictions.shape[1] ).item()  # Accuracy = TP+TN / ALL
        return acc
    
        # TO DO: Devo usare la confusion matrix????????? 
    
    def train(self, train_set, validation_set, lr, epochs, vocabulary):
            
            criterion = nn.CrossEntropyLoss(ignore_index=self.padding_index,reduction="sum").cuda() if self.device.type == "cuda"  \
                                                else nn.CrossEntropyLoss(ignore_index=0,reduction="sum")
            
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

                for images,captions_training_ids,captions_target_ids in train_set:
                    optimizer.zero_grad() 
                    
                    batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size')
                    epoch_num_train_examples += batch_num_train_examples
                    
                    
                    images = images.to(self.device)
                    captions_training_ids = captions_training_ids.to(self.device) # captions > (B, L)
                    captions_target_ids = captions_target_ids.to(self.device) # captions > (B, |L|-1) without end token

                    # computing the network output on the current mini-batch
                    features = self.C(images)
                    outputs = self.R(features, captions_training_ids) # outputs > (B, L, |V|); 
                    
                    # (B, L, |V|) -> (B * L, |V|) and captions > (B * L)
                    loss = criterion(outputs.reshape((-1,outputs.shape[2])), captions_target_ids.reshape(-1))
                    
                    # computing gradients and updating the network weights
                    loss.backward()  # computing gradients
                    optimizer.step()  # updating weights
                    
                    # with torch.no_grad():
                    #     self.C.eval()
                    #     self.R.eval()
                    #     features = self.C(images)
                    #     import random
                    #     numb = random.randint(0,2)
                    #     caption = self.R.generate_caption(features[numb],30)
                    #     print(vocabulary.rev_translate(captions_target_ids[numb]))
                    #     print(vocabulary.rev_translate(caption[0]))
                    #     self.C.train()
                    #     self.R.train()
                    
                    with torch.no_grad():
                        self.C.eval()
                        self.R.eval()
                        
                        # Compute captions as ids for all the training images
                        projections = self.C(images)
                        
                        captions_output = torch.zeros((projections.shape[0],captions_target_ids.shape[1])).to(torch.device(device))
                        
                        for idx,projection in enumerate(range(projections.shape[0])):
                            _caption_no_pad = self.R.generate_caption(projections[idx],captions_target_ids.shape[1])
                            captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                            # Fill the remaining portion of caption eventually with zeros
                            # Accuracy is not altered since if the length of caption is smaller than the captions_target_ids(padded), feed it with PAD is valid.
    
                        captions_output_padded = captions_output.type(torch.int32).to(torch.device(device)) # From list of tensors to tensors
                        
                        # computing performance
                        batch_train_acc = self.__accuracy(captions_output_padded.squeeze(1), captions_target_ids)

                        # accumulating performance measures to get a final estimate on the whole training set
                        epoch_train_acc += batch_train_acc * batch_num_train_examples

                        # accumulating other stats
                        epoch_train_loss += loss.item() * batch_num_train_examples
                        self.C.train()
                        self.R.train()
                        
                        # printing (mini-batch related) stats on screen
                        print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_acc))
                        
                val_acc = self.eval_classifier(validation_set)

                # saving the model if the validation accuracy increases
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = e + 1
                    self.save("CaRNetvHC")

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
            for images,_,captions_validation_target_ids in data_set:
                images = images.to(self.device)
                
                captions_validation_target_ids = captions_validation_target_ids.to(self.device)

                projections = self.C(images)
                        
                captions_output = torch.zeros((projections.shape[0],captions_validation_target_ids.shape[1])).to(torch.device(device))
                
                for idx,projection in enumerate(range(projections.shape[0])):
                    _caption_no_pad = self.R.generate_caption(projections[idx],captions_validation_target_ids.shape[1])
                    captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                    # Fill the remaining portion of caption eventually with zeros
                    # Accuracy is not altered since if the length of caption is smaller than the captions_target_ids(padded), feed it with PAD is valid.

                captions_output_padded = captions_output.type(torch.int32).to(torch.device(device)) # From list of tensors to tensors
                
                # computing performance
                acc = self.__accuracy(captions_output_padded.squeeze(1), captions_validation_target_ids)

        if training_mode_originally_on:
            self.C.train()  # restoring the training state, if needed
            self.R.train()
        return acc
# Example of usage
if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from Dataset import MyDataset
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset/flickr30k_images/", percentage=8)
    v = Vocabulary(ds,reload=True) 
    dc = ds.get_fraction_of_dataset(percentage=70)
    df = ds.get_fraction_of_dataset(percentage=30)
    # use dataloader facilities which requires a preprocessed dataset
       
    
    dataloader_training = DataLoader(dc, batch_size=100,
                        shuffle=True, num_workers=12, collate_fn = lambda data: ds.pack_minibatch_training(data,v))
    
    dataloader_evaluation = DataLoader(df, batch_size=50,
                        shuffle=True, num_workers=12, collate_fn = lambda data: ds.pack_minibatch_evaluation(data,v))
    
    net = CaRNet1(512,0,len(v.word2id.keys()),v.embeddings.shape[1],"cpu")
    net.load("CaRNetvHC")
    net.train(dataloader_training,dataloader_evaluation,1e-3,500,v)
