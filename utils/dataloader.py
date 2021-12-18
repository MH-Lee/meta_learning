import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MiniImagenet(Dataset):
    def __init__(self, path, N=5 ,K=2, Q=15, mode='train', total_iter=60000, transform=False):
        super(MiniImagenet, self).__init__()
        self.total_iter = total_iter
        self.N_way = N
        self.K_support = K
        self.Q_query = Q
        self.path = path
        if transform:
            self.transform = transforms.Compose([lambda x:x/255.0])
        else:
            self.transform = None
        self.mode = mode
        pickle_file = os.path.join(self.path,\
                                  'mini-imagenet-cache-' + mode + '.pkl')
        pkl_file = pd.read_pickle(pickle_file)
        img = pkl_file['image_data']
        self.label = pkl_file['class_dict']
        self.X = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        self.y = np.ones(len(self.X))
        
        self.class_idx = {v:k for k,v in enumerate(self.label.keys())}
        
        for class_name, idxs in self.label.items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]
        self.sx, self.qx, self.sy, self.qy = self.few_shot_sampler()
                
    def __getitem__(self, idx):
        support_data = self.X[self.sx[idx]]
        query_data = self.X[self.qx[idx]]
        if self.transform:
            support_data = self.transform(support_data)
            query_data = self.transform(query_data)
        return support_data, torch.LongTensor(self.sy[idx]), query_data, torch.LongTensor(self.qy[idx])
    
    def __len__(self):
        return len(self.sx)
    
    def few_shot_sampler(self):
        total_support_x, total_support_y = list(), list()
        total_query_x, total_query_y = list(), list()
        for _ in tqdm(range(self.total_iter)):
            task_support_x, task_support_y = list(), list()
            task_query_x, task_query_y = list() , list()
            class_sample = np.random.choice(list(self.class_idx.keys()), self.N_way, False)
            new_cls_label = list(range(0, self.N_way))
            np.random.shuffle(new_cls_label)
            for idx, cls_idx in enumerate(class_sample):
                sample_idx = np.random.choice(self.label[cls_idx], self.K_support+self.Q_query, False)
                support_sample = list(sample_idx[:self.K_support])
                query_sample = list(sample_idx[self.K_support:])
                task_support_x.extend(support_sample)
                task_query_x.extend(query_sample)
                task_support_y.extend([new_cls_label[idx]]*self.K_support)
                task_query_y.extend([new_cls_label[idx]]*self.Q_query)
            total_support_x.append(task_support_x)
            total_query_x.append(task_query_x)
            total_support_y.append(task_support_y)
            total_query_y.append(task_query_y)
        return total_support_x, total_query_x, total_support_y, total_query_y

if __name__ == '__main__':
    root_path = './datasets/miniimagenet/pkl_file/' 
    # transform = transforms.Compose([lambda x:x/255.0])
    dataset = MiniImagenet(path=root_path)