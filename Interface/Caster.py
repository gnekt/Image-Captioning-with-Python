from abc import ABC, abstractmethod
import torch

class ABCCaster(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def cast(self, object_i):
        pass



class CastTorchTensorToDataloader(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def cast(self, object_i):
        target = torch.tensor(df['Targets'].values)
        features = torch.tensor(df.drop('Targets', axis = 1).values)

        train = data_utils.TensorDataset(features, target)
        train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    
    
    
class CastDataframeToTorchTensor(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def cast(self, object_i):
        pass
    
    
    
class Caster():
    TorchTensorToDataloader = CastTorchTensorToDataloader
    DataframeToTorchTensor = CastDataframeToTorchTensor