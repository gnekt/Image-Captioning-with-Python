from .Encoder.CResNet50 import CResNet50
from .Encoder.CResNet50Attention import CResNet50Attention
from .Decoder.RNetvHC import RNetvHC
from .Decoder.RNetvI import RNetvI
from .Decoder.RNetvH import RNetvH
from .Decoder.RNetvHCAttention import RNetvHCAttention
from .CaRNet import CaRNet
from .Attention.SoftAttention import SoftAttention
from enum import Enum

# Open source is a development methodology; free software is a social movement. 
# - Richard Stallman

####### How to continue implementation?
## Everyone is free of enrich this library, remember to follow the IInterface.py for each type of Elements
## At the end of your code session, add your element to the factory. 
## Follow the rule:

class Attention(Enum):
    """
        Attention type list.
    """
    Attention = 0
    
def FactoryAttention(attention: Attention):
    """ Attention Factory 

    Args:
        attention (Attention): 
            The expected attention to produce

    Raises:
        NotImplementedError: Raise when external ask for an implementation that is not covered yet.

    Returns:
        (IAttetion):   
            A Class reference
    """
    if attention == Attention.Attention:
        return SoftAttention
    raise NotImplementedError("This attention model is not implemented yet")

####################################################################

class Encoder(Enum):
    """
        Encoder type list.
    """
    CResNet50 = 0
    CResNet50Attention = 1
    
def FactoryEncoder(encoder: Encoder):
    """ Encoder Factory 

    Args:
        encoder (Encoder): 
            The expected encoder to produce

    Raises:
        NotImplementedError: Raise when external ask for an implementation that is not covered yet.

    Returns:
        (IEncoder):   
            A Class reference
    """
    if encoder == Encoder.CResNet50:
        return CResNet50
    if encoder == Encoder.CResNet50Attention:
        return CResNet50Attention
    raise NotImplementedError("This encoder is not implemented yet")

####################################################################

class Decoder(Enum):
    """
        Decoder type list.
    """
    RNetvI = 0
    RNetvH = 1
    RNetvHC = 2
    RNetvHCAttention = 3 
    
def FactoryDecoder(decoder: Decoder):
    """ Decoder Factory 

    Args:
        decoder (Decoder): 
            The expected decoder to produce

    Raises:
        NotImplementedError: Raise when external ask for an implementation that is not covered yet.

    Returns:
        (IDecoder):   
            A Class reference
    """
    if decoder == decoder.RNetvI:
        return RNetvI
    if decoder == decoder.RNetvH:
        return RNetvH
    if decoder == decoder.RNetvHC:
        return RNetvHC
    if decoder == decoder.RNetvHCAttention:
        return RNetvHCAttention
    raise NotImplementedError("This decoder is not implemented yet")

#####################################################################

class NeuralNet(Enum):
    """
        NeuralNet type list.
    """
    CaRNet = 0

def FactoryNeuralNet(net: NeuralNet):
    """ NeuralNet Factory 

    Args:
        net (NeuralNet): 
            The expected neural net to produce

    Raises:
        NotImplementedError: Raise when external ask for an implementation that is not covered yet.

    Returns:
        (NeuralNet):   
            A Class reference
    """
    if net == NeuralNet.CaRNet:
        return CaRNet
    raise NotImplementedError("This neural net is not implemented yet")