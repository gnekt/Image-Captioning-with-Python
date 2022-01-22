import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) # attach a linear layer ()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from PreProcess import PreProcess
    from Dataset import MyDataset
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset")
    df = ds.get_fraction_of_dataset(percentage=10)
    print("pippo")
    
    # use dataloader facilities which requires a preprocessed dataset
    v = Vocabulary(verbose=True)    
    df_pre_processed,v_enriched = PreProcess.DatasetForTraining.process(dataset=df,vocabulary=v)
    
    dataloader = DataLoader(df, batch_size=4,
                        shuffle=False, num_workers=0, collate_fn=df.pack_minibatch)
    
    encoder = EncoderCNN(50)
    
    for images,captions in dataloader:
        print(encoder(images))