import torch 
from torch import nn
import torch.nn.functional as F


class CosineComponent(nn.Module):
    
    def __init__(self, embedding_size, n_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embedding_size, n_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm

def arcface_loss(cosine, target, output_classes, m=.4):
        
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    arcosine = cosine.arccos()
    arcosine += F.one_hot(target, num_classes=output_classes) * m
    cosine2 = arcosine.cos()
    
    return F.cross_entropy(cosine2, target)

    
class DML(nn.Module):
    def __init__(self, input_size, embedding_size, n_classes, dropout=0.1) -> None:
        super(DML, self).__init__()
        
        self.ffwd_model = nn.Sequential(
            nn.Linear(input_size, 1024), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(1024, embedding_size)
        )
        
        self.last_layer = CosineComponent(embedding_size=embedding_size, n_classes=n_classes)
        
    def get_embeddings(self, batch):
        return self.ffwd_model(batch)

    def forward(self, batch):
        out = self.ffwd_model(batch)
        return self.last_layer(out)