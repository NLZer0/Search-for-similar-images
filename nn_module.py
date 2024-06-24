import torchvision
from torchvision.models import ResNet18_Weights
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class CosineComponent(nn.Module):
    
    def __init__(self, embedding_size:int, n_classes:int):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embedding_size, n_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm

def arcface_loss(cosine:torch.Tensor, target:torch.Tensor, n_classes:int, m:float=.4):
        
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    arcosine = cosine.arccos()
    arcosine += F.one_hot(target, num_classes=n_classes) * m
    cosine2 = arcosine.cos()
    
    return F.cross_entropy(cosine2, target)

    
class DML(nn.Module):
    def __init__(self, embedding_size:int, n_classes:int, dropout:float=0.1) -> None:
        super(DML, self).__init__()
        
        self.pretrain_resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for param in self.pretrain_resnet.parameters():
            param.requires_grad = False

        modules = list(self.pretrain_resnet.children())[:-1]
        self.pretrain_resnet = torch.nn.Sequential(*modules)

        input_embedding_size = 512
        self.ffwd_model = nn.Sequential(
            nn.Linear(input_embedding_size, 1024), 
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(1024, 256), 
            nn.BatchNorm1d(256),
            nn.LeakyReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(256, embedding_size)
        )
        self.last_layer = CosineComponent(embedding_size=embedding_size, n_classes=n_classes)
        
    def get_pretrain_embeddings(self, batch:torch.Tensor):
        return self.pretrain_resnet(batch).squeeze(dim=(2,3))
        
    def get_embeddings(self, batch:torch.Tensor):
        out = self.get_pretrain_embeddings(batch)
        return self.ffwd_model(out)

    def forward(self, batch:torch.Tensor):
        out = self.get_embeddings(batch)
        return self.last_layer(out)