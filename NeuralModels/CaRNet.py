#####################################################
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
    """
        The ConvolutionalandRecurrentNet (CaRNet).
        CaRNet works with a Residual NeuralNet with 50layers (ResNet50) with the last layer removed.
        In CaRNet it supports 3 types of LSTM:
        - vI: the features extracted from the image are provided as input with <START> token
        - vH: the features extracted from the image becames the hidden state at t_0
        - vHC: the features extracted from the image becames both the hidden and cell state at t_0
        
        When it is flavoured with Attention, it becames a ConvolutionalAttentionRecurrentNet (CARNet).
        CARNet works with a Residual NeuralNet with 50layers (ResNet50) with the last convolutional layer exposed.
        For now support only 1 type of LSTM:
        - vHC
    """
    
    def __init__(self, encoder: IEncoder, decoder: IDecoder, net_name: str, encoder_dim: int, hidden_dim: int, padding_index: int, vocab_size: int, embedding_dim: int, attention: IAttention = None, attention_dim: int = 1024, device: str = "cpu"):
        """Create the C[aA]RNet 

        Args:
            encoder (IEncoder): 
                The encoder to use.
                
            decoder (IDecoder): 
                The decoder to use.
                
            net_name (str): 
                Name of the Neural Network.
                
            encoder_dim (int): 
                The dimensionality of the features vector extracted from the image.
                
            hidden_dim (int): 
                The Capacity of the LSTM Cell.
                
            padding_index (int): 
                The index of the padding id, given from the vocabulary associated to the dataset.
                
            vocab_size (int)): 
                The size of the vocabulary associated to the dataset.
                
            embedding_dim (int): 
                Size associated to the input of the LSTM cell.
                
            attention (IAttention, optional): (Default is None)
                The attention if Provided.
                
            attention_dim (int, optional): (Default is 1024)
                Size of the attention layer, used only if attention is not None.
                
            device (str, optional): 
                The device on which the net does the computation. Defaults to "cpu".
        """

        super(CaRNet, self).__init__()
        self.padding_index = padding_index
        self.device = torch.device(device)
        self.name_net = net_name
        
        # Define Encoder and Decoder
        self.C = encoder(encoder_dim = encoder_dim, device = device)
        self.R = None
        
        # Take the attention in consideration
        self.attention = False
        
        if attention is not None: # I know..some skilled dev. will hate me for this if-else statement. Forgive ME.
            self.attention = True
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device, attention(self.C.encoder_dim, hidden_dim, attention_dim))
        else:
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device)

        # Check if the Recurrent net was initialized oth. we are in error state.
        if self.R is None:
            raise ValueError("Could not create the Recurrent network.")
        
        # Send both net to the defined device -> cpu or gpu 
        self.C.to(self.device)
        self.R.to(self.device)
    
    def switch_mode(self, mode: str) -> bool: 
        """ Change the working modality of the net among "training" or "evaluation".

        Args:
            mode (str): 
                New mode of work, "training" | "evaluation"

        Returns:
            bool: 
                If True the state is correctly changed, oth. not.
        """
        # Q. Why no control if they already stay in the wanted state?
        # A. Increase the condition may lead to more than expected case to control. Avoid IfElse community addicted :) 
        if mode == "training":
            self.C.train()  # switch to training state
            self.R.train()
            return True
        
        if mode == "evaluation":
            self.C.eval() # switch to evaluation state
            self.R.eval()
            return True
        return False
    
    def save(self, file_path: str) -> bool:
        """Save the net in non-volatile memory

        Args:
            file_name (str): Relative path to save the net. Ex. "home/pippo/saved"

        Returns:
            bool: If True: Net saved correctly. False otherwise.
        """
        try:
            # Name_type_encoderdim_embeddingdim_hiddendim_attentiondim
            torch.save(self.C.state_dict(), f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention == True else 0}_C.pth")
            torch.save(self.R.state_dict(), f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention == True else 0}_R.pth")
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
        self.C.load_state_dict(torch.load(f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention == True else 0}_C.pth", map_location=self.device))
        self.R.load_state_dict(torch.load(f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention == True else 0}_R.pth", map_location=self.device))
    
    def forward(self, images: torch.tensor, captions: torch.tensor) -> torch.tensor:
        """Provide images to the net for retrieve captions

        Args:
            images (torch.tensor): `(Batch Size, Channels, Width, Height)`
                The images of the batch.
                
            captions (torch.tensor): `(Batch Size, Max_Captions_Length)`. 
                ASSUMPION: The captions are padded with <PAD> Token

        Returns:
            (torch.tensor): `(batch_size, max_captions_length, vocab_size)`
                The output of each time step from t_1 to t_N.
                    REMARK <START> token is provided as output at t_0
        """
        features = self.C(images)
        return self.R(features, captions)

    def __accuracy(self, outputs: torch.tensor, labels: torch.tensor, captions_length: List[int]) -> float:
        """Evaluate the accuracy of the Net with Jaccard Similarity.
                Assumption: outputs and labels have same shape and already padded.

        Args:
            outputs (torch.tensor): `(batch_dim, MAX_CAPTION_LENGTH)`
                The captions generated from the net.
            labels (torch.tensor): `(batch_dim, MAX_CAPTION_LENGTH)` 
                The Real captions.
            captions_length (list): 

        Returns:
            float: The accuracy of the Net
        """
        

        # computing the accuracy with Jaccard Similarity, pytorch unique facility has bugs with cuda....it can be done "a manella" :)
        # from python 3.9 you could use the package torchmetrics
        # from torchmetrics import JaccardIndex
        # intersection_over_union = JaccardIndex(num_classes=self.R.vocab_size).cuda() if self.device.type != "cpu" else JaccardIndex(num_classes=self.R.vocab_size)
        # return intersection_over_union(outputs, labels)
        outputs = np.array(list(map(lambda output: np.unique(output), outputs.cpu()))) # Remove duplicate from each caption
        labels = np.array(list(map(lambda label: np.unique(label), labels.cpu()))) # Remove duplicate from each caption
        
        unions = list(map(lambda index: len(np.union1d(outputs[index],labels[index])), range(labels.shape[0])))
        intersections = list(map(lambda index: len(np.intersect1d(outputs[index],labels[index])), range(labels.shape[0])))
        return torch.mean(torch.tensor(intersections).type(torch.float)/torch.tensor(unions).type(torch.float), axis=0)
    
    
    def train(self, train_set: MyDataset, validation_set: MyDataset, lr: float, epochs: int, vocabulary: Vocabulary):
        """Train the net

        Args:
            train_set (MyDataset): 
                The associate training set.
                
            validation_set (MyDataset): 
                The associate validation set.
                
            lr (float): 
                The learning rate.
                
            epochs (int): 
                The number of epochs.
                
            vocabulary (Vocabulary): 
                The vocabulary associate to the Dataset
        """
        
        # Initialize Loss: CrossEntropyLoss -> Softmax + NegativeLogLikelihoodLoss 
        # Q. Why ignore_index is setted to <START> instead of <PAD>?
        # A. In the training, both output of the CaRNet and Target is a padded tensor, but when we compute the loss it will evaluate the tensor with pack_padded_sequence.
        #       And since <START> token is hardcoded as output at t_0 and it is contained into the Target we could avoid the computation of loss on it, since will be 1.                     
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<START>"],reduction="sum").cuda() if self.device.type == "cuda"  \
                                            else nn.CrossEntropyLoss(ignore_index=vocabulary.predefined_token_idx()["<START>"],reduction="sum")
        
        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data
        best_epoch = -1  # the epoch in which the best accuracy above was computed
        
        # ensuring the classifier is in 'train' mode (pytorch)
        self.switch_mode("training")

        # creating the optimizer
        optimizer = torch.optim.Adam(list(self.R.parameters()) + list(self.C.parameters()), lr)

        # loop on epochs
        for e in range(0, epochs):

            # epoch stats (computed by accumulating mini-batch stats)
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0

            for images,captions_ids,captions_length in  train_set:
                optimizer.zero_grad() 
                
                batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size') -> last batch truncated
                epoch_num_train_examples += batch_num_train_examples
                
                # Send data to the appropriate device
                images = images.to(self.device)
                captions_ids = captions_ids.to(self.device)
                captions_length = captions_length.to(self.device)
                
                # computing the network output on the current mini-batch
                # If Attention is on:
                # In: (batch_dim, channels, height, width) Out: (batch_dim,H_portions, W_portions, encoder_dim) 
                # Else:
                # In: (batch_dim, channels, height, width) Out: (batch_dim, encoder_dim)
                # Retrieve Features for each image
                features = self.C(images)
                
                # Check if attention is provided, if yes the output will change accordly for fitting doubly stochastic gradient
                if self.attention == False: # I know..some skilled dev. will hate me for this if-else statement. Forgive ME.
                    outputs, _ = self.R(features, captions_ids, captions_length) # outputs > (B, L, |V|); 
                else:
                    outputs, _, alphas =  self.R(features, captions_ids, captions_length)
                
                outputs = pack_padded_sequence(outputs, captions_length.cpu(), batch_first=True)  #(Batch, MaxCaptionLength, |Vocabulary|) -> (Batch * CaptionLength, |Vocabulary|)
                
                targets = pack_padded_sequence(captions_ids, captions_length.cpu(), batch_first=True) #(Batch, MaxCaptionLength) -> (Batch * CaptionLength)
                
                loss = criterion(outputs.data, targets.data)
                
                # Doubly stochastic gradient if attention is ON
                if self.attention == True:
                    loss += float(torch.sum((
                                        0.5 * torch.sum((
                                                            (1 - torch.sum(alphas, dim=1,keepdim=True)) ** 2 # caption_length sum
                                                        ), dim=2, keepdim=True) # alpha_dim sum
                                    ), dim=0).squeeze(1)) # batch_dim sum
                    
                # computing gradients and updating the network weights
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                # Training set accuracy evaluation
                with torch.no_grad():
                    self.switch_mode("evaluation")
                    
                    # computing the network output on the current mini-batch
                    # If Attention is on:
                    # In: (batch_dim, channels, height, width) Out: (batch_dim,H_portions, W_portions, encoder_dim) 
                    # Else:
                    # In: (batch_dim, channels, height, width) Out: (batch_dim, encoder_dim)
                    # Retrieve Features for each image
                    projections = self.C(images)
                    
                    # Create a padded tensor manually
                    captions_output = torch.zeros((projections.shape[0],captions_ids.shape[1])).to(self.device)
                    
                    for idx, _ in enumerate(range(projections.shape[0])):
                        # OUT: (1, CAPTION_LENGTH)
                        if self.attention == True:
                            _caption_no_pad, _ = self.R.generate_caption(projections[idx].unsqueeze(0),captions_ids.shape[1]) # IN: ((1, H_portions, W_portions, encoder_dim), 1)
                        else:
                            _caption_no_pad = self.R.generate_caption(projections[idx].unsqueeze(0),captions_ids.shape[1]) # IN: ((1, encoder_dim), 1)
                        # Add for each batch element the caption. The surplus element are already feeded with zeros
                        captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                        

                    captions_output_padded = captions_output.type(torch.int32).to(self.device) # Out: (batch_dim, MAX_CAPTION_LENGTH)
                    
                    # computing performance
                    batch_train_acc = self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length)

                    # accumulating performance measures to get a final estimate on the whole training set
                    epoch_train_acc += batch_train_acc * batch_num_train_examples

                    # accumulating other stats
                    epoch_train_loss += loss.item() * batch_num_train_examples
                    
                    self.switch_mode("training")
                    
                    # printing (mini-batch related) stats on screen
                    print(f"  mini-batch:\tloss={loss.item():.4f}, tr_acc={batch_train_acc:.5f}")
            
            # Evaluate the accuracy of the validation set
            val_acc = self.__eval_net(validation_set,vocabulary)

            # # saving the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1                
                self.save("./.saved")

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(f"epoch={e + 1}/{epochs}:\tloss={epoch_train_loss:.4f}, tr_acc={epoch_train_acc / epoch_num_train_examples:.5f}, val_acc={val_acc:.5f}, {'BEST!' if best_epoch == e+1 else ''}")

    def __eval_net(self, data_set, vocabulary):
        """ Dunder method: Evaluate the validation set

        Args:
            data_set (MyDataset): 
                The associate data set.
                
            vocabulary (Vocabulary): 
                The vocabulary associate to the Dataset

        Returns:
            (int):
                Accuracy on given dataset
        """
        
        self.switch_mode("evaluation")  # enforcing evaluation mode
        with torch.no_grad():  # keeping off the autograd engine
            _images = None
            # loop on mini-batches to accumulate the network outputs (creating a new iterator)
            for images,captions_ids,captions_length  in data_set:
                images = images.to(self.device)
                
                captions_ids = captions_ids.to(self.device)
                
                # If Attention is on:
                # In: (batch_dim, channels, height, width) Out: (batch_dim,H_portions, W_portions, encoder_dim)
                # Else:
                # In: (batch_dim, channels, height, width) Out: (batch_dim, encoder_dim) 
                # Retrieve Features for each image
                projections = self.C(images) 
                
                # Create a padded tensor manually
                captions_output = torch.zeros((projections.shape[0],captions_ids.shape[1])).to(self.device)
                
                for idx, _ in enumerate(range(projections.shape[0])):
                    # OUT: (1, CAPTION_LENGTH)
                    if self.attention == True:
                        _caption_no_pad, _ = self.R.generate_caption(projections[idx].unsqueeze(0),captions_ids.shape[1]) # IN: ((H_portions, W_portions, encoder_dim), 1)
                    else:
                        _caption_no_pad = self.R.generate_caption(projections[idx].unsqueeze(0),captions_ids.shape[1]) # IN: ((1, encoder_dim), 1)
                   # Add for each batch element the caption. The surplus element are already feeded with zeros
                    captions_output[idx,:_caption_no_pad.shape[1]] = _caption_no_pad
                
                # Pick the 1st image of the last batch for printing out the result 
                _image = images[0]
                captions_output_padded = captions_output.type(torch.int32).to(self.device) # Out: (batch_dim, MAX_CAPTION_LENGTH)
                
                # computing performance
                acc = self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length)
            
            self.eval(_image,vocabulary)
        self.switch_mode("training")
        
        return acc
    
    def __generate_image_caption(self, image: torch.Tensor, vocabulary: Vocabulary, image_name: str = "caption.png"):
        """ Genareate an image with caption.

        Args:
            image (torch.Tensor): `(channels, height, width)`
                The tensorial representation of the image in resnet50 form.
                
            vocabulary (Vocabulary): 
                The vocabulary associated to the dataset.
                
            image_name (str, optional): Defaults to "caption.png".
                The image of the generated file
        """
        self.switch_mode("evaluation")  # enforcing evaluation mode
        
        # If Attention is on:
        # Out: 1st step (batch_dim,H_portions, W_portions, encoder_dim) -> 2nd step (batch_dim, H_portions * W_portions, encoder_dim) 
        # Else:
        # Out: (1, encoder_dim) 
        features = self.C(image.unsqueeze(0))
        
        if self.attention == True:
            features = features.reshape(1,-1,self.C.encoder_dim) 
            caption, alphas = self.R.generate_caption(features,MAX_CAPTION_LENGTH)
        else:
            caption = self.R.generate_caption(features,MAX_CAPTION_LENGTH)
    
        # Generate image caption
        caption = vocabulary.rev_translate(caption[0])
        
        # Adjust the color of the image wrt the transform operation of the resnet50
        image[0] = image[0] * 0.229
        image[1] = image[1] * 0.224 
        image[2] = image[2] * 0.225 
        image[0] += 0.485 
        image[1] += 0.456 
        image[2] += 0.406
        
        # Swap color channels
        image = image.permute((1,2,0)) # IN: (height, width, channels)
        
        # If attention is ON perform the evaluation of attention over the immage
        if self.attention == True:
            self.__generate_image_attention(image, caption, alphas)

        plt.figure(figsize=(15, 15))
        plt.imshow(image.cpu())
        plt.title(caption)
        plt.savefig("caption.png")
        plt.close()
        
        self.switch_mode("training")
        
    def __generate_image_attention(self, image: torch.tensor, caption, alphas, image_name: str = "attention.png"):
        """Perform the evaluation of the attention over the image.

        Args:
            image (torch.Tensor): 
                The tensorial representation of the image.
                
            caption (list(str)): 
                The caption.
                
            alphas (torch.Tensor): 
            
            image_name (str, optional): Defaults to "attention.png".
                The image of the generated file
        """
        self.switch_mode("evaluation") 
        
        fig = plt.figure(figsize=(15, 15))
        _caption_len = len(caption)
        for t in range(_caption_len):
            # from 49 element to 7x7
            _att = alphas[t].reshape(self.attention.number_of_splits,self.attention.number_of_splits)
            
            # Add a subplot accordly to the word in caption position
            ax = fig.add_subplot(_caption_len//2, _caption_len//2, t+1)
            
            ax.set_title(f"{caption[t]}", fontsize=12)
            
            img = ax.imshow(image.cpu())
            
            # Add attention layer
            ax.imshow(_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        plt.tight_layout()
        plt.savefig(image_name)
        plt.close()
        
        self.switch_mode("training")
        
    # Inspiration is taken from this example https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch
    # Thanks ABISHEK BASHYAL :)
    def eval(self, image: object, vocabulary: Vocabulary):
        """Evaluate an image and retrieve the associated caption.

        Args:
            image (PIL.Image.Image or torch.Tensor):  if tensor `(channels, height, width)`
                The image for which it evaluate the caption. 
                
            vocabulary (Vocabulary): 
                The vocabulary.

        Raises:
            ValueError: If the image is not a tensor or an image.
        """
        # enforcing evaluation mode
        self.switch_mode("evaluation")
        
        if isinstance(image, Image.Image):
            operations = transforms.Compose([
                transforms.Resize((MyDataset.image_trasformation_parameter["crop"]["size"], MyDataset.image_trasformation_parameter["crop"]["size"])),  # Crops the given image at the center.
                transforms.ToTensor(),
                transforms.Normalize(mean=MyDataset.image_trasformation_parameter["mean"], std=MyDataset.image_trasformation_parameter["std_dev"])
            ])
            image = operations(image)
        
        if not(isinstance(image,torch.Tensor)): 
            raise ValueError(f"Image is not the expected type, got: {type(image)}.")
        
        self.__generate_image_caption(image,vocabulary)
        
        self.switch_mode("training")
        
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
