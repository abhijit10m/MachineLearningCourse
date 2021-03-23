import torch
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image

class EyeDataset(data.Dataset):
   def __init__(self, xFilePaths, yLabels):

      self.xFilePaths = xFilePaths
      self.yLabels = yLabels
      
      self.xImages = [ Image.open(path) for path in self.xFilePaths ]

      self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[1.0])
      ])

      self.xTensors = [ self.transform(image) for image in self.xImages ]
      self.yTensors = [ torch.tensor( [ float(yValue) ] ) for yValue in self.yLabels ]

   def __len__(self):
      return len(self.xImages)

   def __getitem__(self, index):
      x = self.xTensors[index]
      y = self.yTensors[index]

      return x, y
   
class EyeDatasetNoCache(data.Dataset):
   def __init__(self, xFilePaths, yLabels):

      self.xFilePaths = xFilePaths
      self.yLabels = yLabels
      
      self.transform = transforms.Compose([
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[1.0])
         ])

      self.yTensors = [ torch.tensor( [ float(yValue) ] ) for yValue in self.yLabels ]

   def __len__(self):
      return len(self.xFilePaths)

   def __getitem__(self, index):
      x = self.transform(Image.open(self.xFilePaths[index]))
      y = self.yTensors[index]

      return x, y
