from .Encoder.CResNet50 import CResNet50
from .Encoder.CResNet50Attention import CResNet50Attention
from .Decoder.RNetvHC import RNetvHC
from .Decoder.RNetvI import RNetvI
from .Decoder.RNetvH import RNetvH
from .Decoder.RNetvHCAttention import RNetvHCAttention
from .CaRNet import CaRNet
from .Attention.SoftAttention import SoftAttention
from enum import Enum


class Attention(Enum):
    Attention = 0
    
def FactoryAttention(attention: Attention):
    if attention == Attention.Attention:
        return SoftAttention
    raise NotImplementedError("This attention model is not implemented yet")

####################################################################

class Encoder(Enum):
    CResNet50 = 0
    CResNet50Attention = 1
def FactoryEncoder(encoder: Encoder):
    if encoder == Encoder.CResNet50:
        return CResNet50
    if encoder == Encoder.CResNet50Attention:
        return CResNet50Attention
    raise NotImplementedError("This encoder is not implemented yet")

####################################################################

class Decoder(Enum):
    RNetvI = 0
    RNetvH = 1
    RNetvHC = 2
    RNetvHCAttention = 3 
    
def FactoryDecoder(decoder: Decoder):
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
    CaRNet = 0

def FactoryNeuralNet(net: NeuralNet):
    if net == NeuralNet.CaRNet:
        return CaRNet
    raise NotImplementedError("This neural net is not implemented yet")