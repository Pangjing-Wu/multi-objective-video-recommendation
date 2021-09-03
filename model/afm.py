import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFMLayer(nn.Module):
    '''Attentional Factorization Machine.
    '''

    def __init__(self, field_dims, embed_dim, output_dim, atten_hidden_dim, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, output_dim)
        self.afm = AttentionalFactorizationMachine(embed_dim, output_dim, atten_hidden_dim, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, n_feature)``
        """
        x = x.to(next(self.parameters()).device)
        x = self.linear(x) + self.afm(self.embedding(x))
        return x


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc   = nn.Embedding(sum(field_dims), output_dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array([0, *np.cumsum(field_dims)[:-1]], dtype=np.long)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, n_feature)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim, padding_idx=0)
        self.offsets = np.array([0, *np.cumsum(field_dims)[:-1]], dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, n_feature)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class AttentionalFactorizationMachine(nn.Module):

    def __init__(self, embed_dim, output_dim, atten_hidden_dim, dropout):
        super().__init__()
        self.linear    = nn.Linear(embed_dim, output_dim)
        self.dropout   = dropout
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, atten_hidden_dim),
            nn.ReLU(),
            nn.Linear(atten_hidden_dim, 1),
            nn.Softmax(dim=1),
            nn.Dropout(p=self.dropout)
        )
        

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, n_feature, embed_dim)``
        """
        n_feature = x.shape[1]
        row, col  = list(), list()
        for i in range(n_feature - 1):
            for j in range(i + 1, n_feature):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        # [N, (n_feature - 1)**2, embed_dim]
        inner_product = p * q
        # [N, (n_feature - 1)**2, 1]
        attention = self.attention(inner_product)
        # [N, embed_dim]
        inner_product = torch.sum(inner_product * attention, dim=1)
        inner_product = F.dropout(inner_product, p=self.dropout)
        return self.linear(inner_product)
