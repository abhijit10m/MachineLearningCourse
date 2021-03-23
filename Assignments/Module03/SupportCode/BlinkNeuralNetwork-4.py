import torch

class BlinkNeuralNetworkColab(torch.nn.Module):
    def __init__(self, spec):
        super(BlinkNeuralNetworkColab, self).__init__()
        
        
        self.b0 = torch.nn.BatchNorm2d(1)
        
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=8,
                            kernel_size=3, 
                            stride=3),
            torch.nn.ReLU())
        
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=4,
                            kernel_size=3, 
                            stride=3),
            torch.nn.ReLU())

        self.b1 = torch.nn.BatchNorm1d(num_features=16)

        self.d1 = torch.nn.Dropout()

        self.f1 = torch.nn.Sequential(
            torch.nn.Linear(16, 9),
            torch.nn.ReLU())

        self.d2 = torch.nn.Dropout()

        self.f2 = torch.nn.Sequential(
            torch.nn.Linear(9, 2),
            torch.nn.ReLU())

        self.o  = torch.nn.Sequential(
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid())



    def forward(self, x):

        out = self.b0(x)

        out = self.c1(out)  

        out = self.c2(out)  

        out = out.reshape(out.size(0), -1)

        out = self.d1(out)

        out = self.f1(out)

        out = self.d2(out)

        out = self.f2(out)

        out = self.o(out)

        return out
