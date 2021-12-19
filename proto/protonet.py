import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from common.cnn_models import CNN4Backbone

def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def protonet_train(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc

class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64):
        super().__init__()
        self.encoder = CNN4Backbone(
            hidden_size=hid_dim,
            channels=x_dim,
            max_pool=True,
       )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    
# def conv_block(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )

# # class ConvNet(nn.Module):
# #     def __init__(self, in_channels=3, hid_dim=64, out_dim=64):
# #         super(ConvNet, self).__init__()
# #         self.encoder = nn.Sequential(
# #             conv_block(in_channels, hid_dim),
# #             conv_block(hid_dim, hid_dim),
# #             conv_block(hid_dim, hid_dim),
# #             conv_block(hid_dim, out_dim),
# #         )

# #     def forward(self, x):
# #         x = self.encoder(x)
# #         return x.view(x.size(0), -1)

# def protonet_train(batch, model, metric, criterion,\
#                   ways, shots, device, dist_mode='euclidean'):
#     sx, sy, qx, qy = batch
#     sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
#     NK = ways * shots
#     support_indices = torch.sort(sy)
#     query_indices = torch.sort(qy)
#     sx = sx[support_indices.indices]
#     sy = sy[support_indices.indices]
#     qx = qx[query_indices.indices]
#     qy = qy[query_indices.indices]
#     data = torch.cat((sx,qx),dim=0)
#     embeddings = model(data)
#     support = embeddings[:NK]
#     query = embeddings[NK:]
#     proto = support.reshape(ways, shots, -1).mean(dim=1)
#     logits = metric(query, proto, mode=dist_mode)
#     loss = criterion(logits, qy.long())
#     acc = accuracy(logits, qy.long())
#     return loss, acc

# def distance(query, proto, mode='euclidean'):
#     n = query.size(0)
#     m = proto.size(0)
#     d = query.size(1)
#     if d != proto.size(1):
#         raise Exception
#     x1 = query.unsqueeze(1).expand(n, m, -1)
#     x2 = proto.unsqueeze(0).expand(n, m, -1)
    
#     if mode == 'euclidean':
#         logits = -((x1 - x2)**2).sum(dim=2)
#     else:
#         cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         logits = -cos(x1, x2)
#     return logits
