import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import *
from data.dtype import *
from data.preprocessing import *
from data.generator import BehaviorDataset
from model.dnn import MultiTaskMLP
from model.afm import AFMLayer
from model.loss import FocalLoss

torch.set_default_tensor_type('torch.DoubleTensor') 
device = torch.device('cuda:0')


lr = 1e-4
save_dir  = './results/submission.csv'
model_file = os.path.join(dnn['savedir'], f'40.pth')


class DIGIXNet(nn.Module):

    def __init__(self, lr):
        super().__init__()
        self.afm = AFMLayer(**dnn['afm'])
        self.match_net  = MultiTaskMLP(**dnn['mtmlp']).to(device)
        self.watch_loss = FocalLoss(**dnn['watch_loss'])
        self.share_loss = FocalLoss(**dnn['share_loss'])
        self.optimizer  = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.afm(x)
        
    def get_watch(self, x):
        x = self.forward(x)
        return self.match_net.get_watch(x)

    def get_share(self, x):
        x = self.forward(x)
        return self.match_net.get_share(x)


def test(model, dataloader):
    y1, y2 = list(), list()
    for x in dataloader:
        with torch.no_grad():
            pred_y1 = model.get_watch(x).argmax(dim=1)
            pred_y2 = model.get_share(x).argmax(dim=1)
        y1 += pred_y1.cpu().numpy().tolist()
        y2 += pred_y2.cpu().numpy().tolist()
    return y1, y2


def main():

    print('--- loading data ---')
    start_time = time.time()
    behav_data = pd.read_csv(dataset['test_behavior_file'], sep=',',
                             usecols=['user_id', 'video_id'],
                             dtype=behav_dtype, low_memory=False)
    user_data  = pd.read_csv(dataset['user_file'], sep='\t', index_col='user_id',
                             dtype=user_dtype, low_memory=False)
    # user feature = [age, gender, country, province, city, city_level, device_name]
    user_data  = user_process(user_data)
    video_data = pd.read_csv(dataset['video_file'], sep='\t', 
                             index_col='video_id', dtype=video_dtype,
                             quoting=csv.QUOTE_NONE, low_memory=False)
    # video feature = [video_release_decade, video_score, video_duration, class_vector]
    video_data = video_process(video_data, **dataset['video'])
    print(f'time cost of loading data: {time.time() - start_time:.2f}s.')

    print('\n--- building dataset ---')
    start_time  = time.time()
    test_data   = BehaviorDataset(user_data, video_data, behav_data, False)
    test_loader = DataLoader(test_data, batch_size=dnn['batch'],
                             shuffle=False, num_workers=20)

    print(f'test batch: {len(test_loader)}.')
    print(f'time cost of building dataset: {time.time() - start_time:.2f}s.')

    model = DIGIXNet(lr=lr).to(device)
    model.load_state_dict(torch.load(model_file)['model'])
    
    print(f'\n--- start test at {time.ctime()} ---')
    y1, y2 = test(model, test_loader)
    df = pd.DataFrame(
        dict(
            user_id=behav_data.user_id,
            video_id=behav_data.video_id,
            watch_label=y1,
            is_share=y2
        )
    )
    df.to_csv(f'{save_dir}', sep=',', index=False)
    print(f'\n--- test completed at {time.ctime()} ---')


if __name__ == '__main__':
    main()