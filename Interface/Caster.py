from abc import ABC, abstractmethod

class ABCCaster(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def caster(self, object_i):
        pass



class CastDataframeToDataloader(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def caster(self, object_i):
        pass
    
    
    
class CastDataframeToTorchTensor(ABC):
    """Abstract class which implements casting interface
    """
    
    @abstractmethod
    def caster(self, object_i):
        pass
    
    
    
class Caster():
    CastDataframeToDataloader = CastDataframeToDataloader
    CastDataframeToTorchTensor = CastDataframeToTorchTensor