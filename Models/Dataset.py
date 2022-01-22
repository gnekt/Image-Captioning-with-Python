from xml.dom import ValidationErr

from Sample import Sample
import os
import pandas as pd 
import torch
import numpy as np 
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class DatasetState(Enum):
    """A dataset could be in 3 possible, mutual exclusive, state:
        - Raw -> Sample are raw, no preprocessing operation performed
        - Training -> All Samples are pre-processed for training
        - Evaluation -> All Samples are pre-processed for evaluation

    Args:
        Enum (int): Raw or Training or Evaluation
    """
    Raw = 0
    Training = 1
    Evaluation = 2
    
# TO-Do
# Aggiungere a README, la modalita` in cui si elabora il dataset e`: ho una cartella il cui contenuto e`: 
#       1) un file result.csv che contiene i dati come il formato gia` definito
#       2) una cartella images nella quale ci sono tutte le immagini, tutte le sottocartelle di images non verranno considerate

class MyDataset(Dataset):
    # The dataset will have this shape
    # | id_sample | sample | dirty |
    # |-----------|--------|-------|
    # |           |        |       |
    # |           |        |       |
    # |           |        |       |
    # 
    # id_sample is an unique identifier of the sample
    # sample is the <Sample> object associated 
    # dirty is boolean and it means: this sample was already taken from the method get_fraction_of_dataset, this implies that externally somebody already taken this sample.
    
    def __init__(self, directory_of_data:str = None, already_computed_dataframe: pd.DataFrame = None, state: DatasetState = DatasetState.Raw):
        """Create a new dataset from source files

        Args:
            directory_of_data (str): [description]
        """
        self.state: DatasetState = state
        if already_computed_dataframe is not None:
            self.dataset = already_computed_dataframe
            return 
        
        if not os.path.exists(directory_of_data):
            raise ValueError(f"{directory_of_data} not Exist!")
        if not os.path.isdir(directory_of_data):
            raise ValueError(f"{directory_of_data} is not a directory!")
        
        _temp_dataset=pd.read_csv(f"{directory_of_data}/results.csv", sep="|", skipinitialspace=True)[["image_name","comment"]].iloc[0:1000,:]
        samples = _temp_dataset.apply(lambda row: Sample(int(row.name)+1,f"{directory_of_data}/images/{row.image_name}",row.comment),axis=1)
        
        self.dataset: pd.DataFrame = pd.DataFrame(list(zip([i for i in range(len(samples))],samples,[False for _ in range(len(samples))])), columns=["id_sample","sample","dirty"])
        
    
    def suffle_data_set(self):
        self.dataset.apply(torch.randperm, axis=0)
    
    def get_fraction_of_dataset(self, percentage: int, also_dirty: bool = False): 
        if not also_dirty:
            _temp_df = self.dataset[self.dataset["dirty"] == False]
        _temp_df = _temp_df.apply(np.random.permutation, axis=0)
        _temp_df_moved = _temp_df.head(int(len(_temp_df)*(percentage/100)))
        _temp_df_copy = _temp_df_moved.copy()
        self.dataset.loc[_temp_df_moved["id_sample"],"dirty"] = True
        return MyDataset(already_computed_dataframe=_temp_df_copy)
        

    def make_dirty(self) -> bool:
        self.dataset["dirty"] = True
    
    def make_clean(self) -> bool:
        self.dataset["dirty"] = False
    
    # torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

    # __len__ so that len(dataset) returns the size of the dataset.
    # __getitem__ to support the indexing such that dataset[i] can be used to get i-ith sample.
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        
        if self.state == DatasetState.Raw:
            raise ValidationErr("The getitem built-in method cannot be executed when the dataset is in a RAW state.\n Please do some preprocessing on it before __getitem__ call.")
        print(f"Dataset state: {self.state}")
        
        sample: Sample = self.dataset.iloc[idx]["sample"]
        image, caption = sample.image, sample.caption
        
        return image,caption 
    
    def pack_minibatch(self, data):
        
        images, captions = zip(*data)
        captions = nn.utils.rnn.pad_sequence(captions, padding_value=0)
        return images,captions
#-------------------------------
# Usage

if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from PreProcess import PreProcess
    ds = MyDataset("./dataset")
    df = ds.get_fraction_of_dataset(percentage=10)
    print("pippo")
    
    # use dataloader facilities which requires a preprocessed dataset
    v = Vocabulary(verbose=True)    
    df_pre_processed,v_enriched = PreProcess.DatasetForTraining.process(dataset=df,vocabulary=v)
    
    dataloader = DataLoader(df, batch_size=4,
                        shuffle=False, num_workers=0, collate_fn=df.pack_minibatch)
    
    for i_batch,[images,captions] in enumerate(dataloader):
        print(i_batch, captions)