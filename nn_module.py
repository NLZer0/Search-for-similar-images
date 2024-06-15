import torchvision
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        
        self.pretrain_resnet = torchvision.models.resnet18(pretrained=True)
        input_embedding_size = self.pretrain_resnet.fc.out_features
        self.pretrain_resnet = torch.nn.DataParallel(self.pretrain_resnet)
        
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
        # self.last_layer = nn.Sequential(
        #     nn.Linear(embedding_size, n_classes),
        #     nn.Softmax(dim=1)
        # )
        
    def get_pretrain_embeddings(self, batch:torch.Tensor):
        with torch.no_grad():
            return self.pretrain_resnet(batch)
        
    def get_embeddings(self, batch:torch.Tensor):
        out = self.get_pretrain_embeddings(batch)
        return self.ffwd_model(out)

    def forward(self, batch:torch.Tensor):
        out = self.get_embeddings(batch)
        return self.last_layer(out)
    
    def get_data_hidden_states(self, data_loader:DataLoader, device:torch.device):
        self.eval()
        self.to(device)
        
        with torch.no_grad():
            image_hidden_states = []
            for batch in data_loader:
                image_hidden_states.append(self.get_embeddings(batch[0].to(device)).cpu())

            image_hidden_states = torch.cat(image_hidden_states)
        
        return image_hidden_states
                    
        
    def get_predict(self, data_loader:DataLoader, device:torch.device):
        self.eval()
        self.to(device)
        
        predict = []
        targets = []
        with torch.no_grad():
            for batch in data_loader:
                predict.append(
                    self(batch[0].to(device))
                )
                targets.append(batch[1])
                
        predict = torch.cat(predict)
        predict /= predict.norm(p=2, dim=1)[:, None]
        targets = torch.cat(targets)
        
        return predict, targets
