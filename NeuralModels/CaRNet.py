#####################################################
## DISCLAIMER: IL CODICE E` SOLO DI TESTING! 
# NON GIUDICARLO, POI LO SISTEMO :)
##
##
##  pip install torch==1.3.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from typing import Tuple,List
from .Dataset import MyDataset
from .Vocabulary import Vocabulary
from .Decoder.IDecoder import IDecoder
from .Encoder.IEncoder import IEncoder
from .Attention.IAttention import IAttention
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from VARIABLE import MAX_CAPTION_LENGTH

class CaRNet(nn.Module):
    
    def __init__(self, encoder: IEncoder, decoder: IDecoder, net_name: str, encoder_dim: int, hidden_dim: int, padding_index: int, vocab_size: int, embedding_dim: int, attention: IAttention = None, attention_dim: int = 1024, device: str = "cpu"):
        """Create the CaRNet 

        Args:
            encoder (IEncoder): The encoder to use
            decoder (IDecoder): The decoder to use
            net_name (str): Name of the Neural Network
            encoder_dim (int): The dimensionality of the features vector extracted from the image
            hidden_dim (int): The Capacity of the LSTM Cell
            padding_index (int): The index of the padding id, given from the vocabulary associated to the dataset
            vocab_size (int)): The size of the vocabulary associated to the dataset
            embedding_dim (int): Size associated to the input of the LSTM cell
            device (str, optional): The device on which the net does the computation. Defaults to "cpu".
        """
            
        super(CaRNet, self).__init__()
        self.padding_index = padding_index
        self.device = torch.device(device)
        
        self.name_net = net_name
        # Define Encoder and Decoder
        self.C = encoder(encoder_dim = encoder_dim, device = device)
        self.R = None
        self.attention = None
        if attention is not None:
            self.attention = attention(self.C.encoder_dim, hidden_dim, attention_dim)
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device, self.attention)
        else:
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device)

        if self.R is None:
            raise ValueError("Could not create the Recurrent network.")
        
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
            torch.save(self.C.state_dict(), f"{file_path}/{self.name_net}_C.pth")
            torch.save(self.R.state_dict(), f"{file_path}/{self.name_net}_R.pth")
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
        self.C.load_state_dict(torch.load(f"{file_path}/{self.name_net}_C.pth", map_location=self.device))
        self.R.load_state_dict(torch.load(f"{file_path}/{self.name_net}_R.pth", map_location=self.device))
    
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
        

        # computing the accuracy with Jaccard Similarity, pytorch unique facility has bugs with cuda....it can be done "a manella" :)
        # from python 3.9 you could use the package torchmetrics
        # from torchmetrics import JaccardIndex
        # intersection_over_union = JaccardIndex(num_classes=self.R.vocab_size).cuda() if self.device.type != "cpu" else JaccardIndex(num_classes=self.R.vocab_size)
        # return intersection_over_union(outputs, labels)
        outputs = np.array(list(map(lambda output: np.unique(output), outputs.cpu())))
        labels = np.array(list(map(lambda label: np.unique(label), labels.cpu())))
        
        unions = list(map(lambda index: len(np.union1d(outputs[index],labels[index])), range(labels.shape[0])))
        intersections = list(map(lambda index: len(np.intersect1d(outputs[index],labels[index])), range(labels.shape[0])))
        return torch.mean(torch.tensor(intersections).type(torch.float)/torch.tensor(unions).type(torch.float), axis=0)
    
    
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
        # Q. Why ignore_index is setted to <START> instead of <PAD>?
        # A. In the training, both output of the CaRNet and Target label start as padded tensor, but when we compute the loss it will evaluate the tensor with pack_padded_sequence.
        #       And since <START> token is hardcoded as output at t_0 we could avoid the computation of loss on it, since will be 0 fover.                     
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<START>"],reduction="sum").cuda() if self.device.type == "cuda"  \
                                            else nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<START>"],reduction="sum")
        
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
                
                # Check if attention is provided, if yes the output will change accordly for fitting doubly stochastic gradient
                if self.attention is None:
                    outputs, outputs_length = self.R(features, captions_ids, captions_length) # outputs > (B, L, |V|); 
                else:
                    outputs, outputs_length, alphas =  self.R(features, captions_ids, captions_length)
                
                outputs = pack_padded_sequence(outputs, captions_length.cpu(), batch_first=True)  #(Batch, MaxCaptionLength, |Vocabulary|) -> (Batch * CaptionLength, |Vocabulary|)
                
                targets = pack_padded_sequence(captions_ids, captions_length.cpu(), batch_first=True) #(Batch, MaxCaptionLength) -> (Batch * CaptionLength)
                
                loss = criterion(outputs.data, targets.data)
                if self.attention is not None:
                    loss += float(torch.sum((
                                        0.5 * torch.sum((
                                                            (1 - torch.sum(alphas, dim=1,keepdim=True)) ** 2 # caption_length sum
                                                        ), dim=2, keepdim=True) # alpha_dim sum
                                    ), dim=0).squeeze(1)) # batch_dim sum
                    
                # computing gradients and updating the network weights
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights
                
                
                
                with torch.no_grad():
                    self.C.eval()
                    self.R.eval()
                    
                    # Compute captions as ids for all the training images
                    projections = self.C(images)
                    
                    captions_output = torch.zeros((projections.shape[0],captions_ids.shape[1])).to(self.device)
                    
                    for idx,projection in enumerate(range(projections.shape[0])):
                        _caption_no_pad, _ = self.R.generate_caption(projections[idx],captions_ids.shape[1])
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
                    print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.5f}".format(loss.item(), batch_train_acc))
                    
            val_acc = self.eval_classifier(validation_set)

            # # saving the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                with torch.no_grad():
                    # self.C.eval()
                    # self.R.eval()
                    # features = self.C(images)
                    # import random
                    # numb = random.randint(0,2)
                    # caption = self.R.generate_caption(features[numb],30)
                    # print(vocabulary.rev_translate(captions_ids[numb]))
                    # print(vocabulary.rev_translate(caption.type(torch.int16)[0]))
                    # self.C.train()
                    # self.R.train()
                    self.eval(images[9], vocabulary)
                
                self.save("/content/drive/MyDrive/Progetti/Neural Networks/.saved")

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.5f}, val_acc={4:.5f}"
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
                    _caption_no_pad, _ = self.R.generate_caption(projections[idx],captions_ids.shape[1])
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
    
    
    
    # Inspiration is taken from this example https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch
    # Thanks ABISHEK BASHYAL :)
    def eval(self, image: Image.Image, vocabulary: Vocabulary, image_is_a_tensor: bool = False):
        # enforcing evaluation mode
        self.C.eval()
        self.R.eval() 
        
        
        # operations = transforms.Compose([
        #         transforms.Resize((MyDataset.image_trasformation_parameter["crop"]["size"], MyDataset.image_trasformation_parameter["crop"]["size"])),  # Crops the given image at the center.
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=MyDataset.image_trasformation_parameter["mean"], std=MyDataset.image_trasformation_parameter["std_dev"])
        # ])
        # image = operations(image)
        
        features = self.C(image.unsqueeze(0)).reshape(1,-1,self.C.encoder_dim)
        
        image[0] = image[0] * 0.229
        image[1] = image[1] * 0.224 
        image[2] = image[2] * 0.225 
        image[0] += 0.485 
        image[1] += 0.456 
        image[2] += 0.406
        
        image = image.permute((1,2,0))
        
        caption, alphas = self.R.generate_caption(features,MAX_CAPTION_LENGTH)
        caption = vocabulary.rev_translate(caption[0])
        
        fig = plt.figure(figsize=(15, 15))
        _caption_len = len(caption)
        for t in range(_caption_len):
            _att = alphas[t].reshape(self.attention.number_of_splits,self.attention.number_of_splits)
            
            ax = fig.add_subplot(_caption_len//2, _caption_len//2, t+1)
            
            ax.set_title(caption[t])
            
            img = ax.imshow(image)
            ax.imshow(_att, cmap='gray', alpha=0.7, extent=img.get_extent())    
        plt.tight_layout()
        plt.savefig("test.png")
        
# Example of usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from FactoryModels import *
    ds = MyDataset("./dataset", percentage=1)
    v = Vocabulary(ds,reload=True) 
    
    # Load Encoder and Decoder models
    decoder = FactoryDecoder(Decoder.RNetvI)
    encoder = FactoryEncoder(Encoder.CResNet50Attention)
    
    dc = ds.get_fraction_of_dataset(percentage=70, delete_transfered_from_source=True)
    df = ds.get_fraction_of_dataset(percentage=30, delete_transfered_from_source=True)
    # use dataloader facilities which requires a preprocessed dataset
       
    
    dataloader_training = DataLoader(dc, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: ds.pack_minibatch_training(data,v))
    
    dataloader_evaluation = DataLoader(df, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: ds.pack_minibatch_evaluation(data,v))
    
    
    net = CaRNet(encoder, decoder, "CaRNetvI",1596,512,0,len(v.word2id.keys()),v.embeddings.shape[1],"cuda:0")
    #net.load("CaRNetvI")
    net.train(dataloader_training,dataloader_evaluation,1e-3,500,v)
