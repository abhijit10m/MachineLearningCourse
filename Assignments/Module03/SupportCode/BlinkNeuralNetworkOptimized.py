import torch
import sys
import logging
import math


logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')


class ConvulationLayerFactory(object):

  # Number of connections = out_channels * (kernel^2  + 1) * ((in_channel - kernel)^2)

    def __init__(self, in_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = math.floor( (self.in_channels - self.kernel_size)/ self.stride) + 1
        logger.info('out_channels %d', self.out_channels)

    def create(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size, 
                            stride=self.stride),
            torch.nn.ReLU())

class AveragePoolingLayerFactory(object):
    def __init__(self, kernel_size, stride):
        self.stride = stride
        self.kernel_size = kernel_size

    def create(self):
        return torch.nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.stride)

class DropoutLayerFactory(object):
    def create(self):
        return torch.nn.Dropout()

class FullyConnectedLayerFactory(object):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def create(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.out_channels),
            torch.nn.Sigmoid())

class OutputLayerFactory(object):
    def __init__(self, in_channels):
        self.in_channels = in_channels

    def create(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, 1),
            torch.nn.Sigmoid())


class BatchNormalizationLayerFactory(object):
    def __init__(self, in_channels):
        self.in_channels = in_channels

    def create(self):
        return torch.nn.BatchNorm2d(num_features = self.in_channels)

class BlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, spec):
        super(BlinkNeuralNetwork, self).__init__()
        self.b1 = BatchNormalizationLayerFactory(1).create()
        # self.c1 = ConvulationLayerFactory(spec['c1']['in_channels'], spec['c1']['kernel'], spec['c1']['stride']).create()
        # self.p1 = AveragePoolingLayerFactory(2,2).create()
        # self.c2 = ConvulationLayerFactory(1, spec['c2']['kernel'], spec['c2']['stride']).create()
        # self.p2 = AveragePoolingLayerFactory(2,2).create()
        self.d1 = DropoutLayerFactory().create()
        self.f1 = FullyConnectedLayerFactory(24 * 24, spec['f1']['out_channles']).create()
        self.f2 = FullyConnectedLayerFactory(spec['f1']['out_channles'], spec['f2']['out_channles']).create()
        self.o  = OutputLayerFactory(spec['f2']['out_channles']).create()

    def forward(self, x):

        logger.debug("input size %s", x.size())
        logger.debug("normalized size %s", x.size())
        out = self.b1(x)
        # out = self.c1(out)      
        # out = self.p1(out)
        # out = self.c2(out)
        # out = self.p2(out)
        out = out.reshape(out.size(0), -1)
        out = self.d1(out)
        out = self.f1(out)
        out = self.f2(out)
        out = self.o(out)
        return out