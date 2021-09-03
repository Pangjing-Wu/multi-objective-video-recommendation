import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import check_dim


class MultiTaskMLP(nn.Module):
    ''' user preference match deep neural network
    Feature -> Linear -> ReLU -> ... -> Linear -> Softmax
    '''
    def __init__(self, input_dim, output_dim,
                 public_hidden_dim, id_hidden_dim,
                 dropout_rate):
        assert isinstance(output_dim, list)
        assert isinstance(public_hidden_dim, list)
        assert isinstance(id_hidden_dim, list)
        super().__init__()
        public_dim = [input_dim] + public_hidden_dim
        watch_dim  = public_hidden_dim[-1:] + id_hidden_dim + output_dim[:-1]
        share_dim  = public_hidden_dim[-1:] + id_hidden_dim + output_dim[-1:]
        # public layers
        self.public = list()
        for i, o in zip(public_dim[:-1], public_dim[1:]):
            self.public.append(nn.Linear(i, o))
            self.public.append(nn.ReLU())
            self.public.append(nn.Dropout(dropout_rate))
        self.public = nn.Sequential(*self.public)
        # watch label classification layers
        self.watch  = list()
        for i, o in zip(watch_dim[:-2], watch_dim[1:-1]):
            self.watch.append(nn.Linear(i, o))
            self.watch.append(nn.ReLU())
            self.watch.append(nn.Dropout(dropout_rate))
        self.watch.append(nn.Linear(watch_dim[-2], watch_dim[-1]))
        self.watch.append(nn.ReLU())
        self.watch = nn.Sequential(*self.watch)
        # share label classification layers
        self.share  = list()
        for i, o in zip(share_dim[:-2], share_dim[1:-1]):
            self.share.append(nn.Linear(i, o))
            self.share.append(nn.ReLU())
            self.share.append(nn.Dropout(dropout_rate))
        self.share.append(nn.Linear(share_dim[-2], share_dim[-1]))
        self.share.append(nn.ReLU())
        self.share = nn.Sequential(*self.share)
        
    def forward(self, x):
        return self.public(x)

    def get_watch(self, x):
        x = self.forward(x)
        x = self.watch(x)
        return F.softmax(x, dim=1)

    def get_share(self, x):
        x = self.forward(x)
        x = self.share(x)
        return F.softmax(x, dim=1)