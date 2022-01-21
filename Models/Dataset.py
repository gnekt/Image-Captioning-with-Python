from Sample import Sample
import os
import pandas as pd 

# TO-Do
# Aggiungere a README, la modalita` in cui si elabora il dataset e`: ho una cartella il cui contenuto e`: 
#       1) un file result.csv che contiene i dati come il formato gia` definito
#       2) una cartella images nella quale ci sono tutte le immagini, tutte le sottocartelle di images non verranno considerate

class Dataset():
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
    
    def __init__(self, directory_of_data:str):
        """Create a new dataset from source files

        Args:
            directory_of_data (str): [description]
        """
        if not os.path.exists(directory_of_data):
            raise ValueError(f"{directory_of_data} not Exist!")
        if not os.path.isdir(directory_of_data):
            raise ValueError(f"{directory_of_data} is not a directory!")
        
        _temp_dataset=pd.read_csv(f"{directory_of_data}/results.csv", sep="|", skipinitialspace=True)[["image_name","comment"]]
        samples = _temp_dataset.apply(lambda row: Sample(int(row.name)+1,f"{directory_of_data}/images/{row.image_name}",row.comment),axis=1)
        print("pippo")
    
    def suffle_data_set(self):
        pass
    
    def get_fraction_of_dataset(self, percentage: int, also_dirty: bool = False): 
        pass
    
    def make_dirty(self) -> bool:
        pass
    
    def make_clean(self) -> bool:
        pass
    
    def __len__(self):
        return self.shape(0)
    

#-------------------------------
# Usage

if __name__ == "__main__":
    ds = Dataset("./dataset")
        