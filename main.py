import argparse
import csv
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import *
from data.dtype import *
from data.preprocessing import *
from data.generator import BehaviorDataset, train_test_split, BalancedResampler
from model.dnn import MultiTaskMLP
from model.afm import AFMLayer
from model.loss import FocalLoss

torch.set_default_tensor_type('torch.DoubleTensor') 
device = torch.device('cuda:0')


epoch = 100
split = 0.99
lr = 3e-4 # learning rate should not be great than 5e-4.


acc = lambda y_pred, y_true: torch.eq(   \
    y_pred[y_true!=0], y_true[y_true!=0]).sum() \
    .float().item() / y_true.shape[0]
mean = lambda x: sum(x) / len(x)


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


def train(model, dataloader):
    watch_accs, share_accs = list(), list()
    watch_losses, share_losses = list(), list()
    watch_count, share_count = Counter(), Counter()
    srart_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        pred_y1 = model.get_watch(x)
        pred_y2 = model.get_share(x)
        y['watch'] = y['watch'].to(device)
        y['share'] = y['share'].to(device)
        watch_loss = model.watch_loss(pred_y1, y['watch'])
        share_loss = model.share_loss(pred_y2, y['share'])
        loss = watch_loss + share_loss
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        watch_losses.append(watch_loss.item())
        share_losses.append(share_loss.item())
        watch_accs.append(acc(pred_y1.argmax(dim=1).detach(), y['watch']))
        share_accs.append(acc(pred_y2.argmax(dim=1).detach(), y['share']))
        watch_count.update(pred_y1.argmax(dim=1).cpu().numpy())
        share_count.update(pred_y2.argmax(dim=1).cpu().numpy())
    time_cost = time.time() - srart_time
    print(f'train - total batch: {len(dataloader):5.0f}, time cost: {time_cost:5.0f}, '
          f'watch acc: {mean(watch_accs):.4f}, share acc: {mean(share_accs):.4f}, '
          f'watch loss: {mean(watch_losses):.4f}, share loss:{mean(share_losses):.4f}.')
    print(f'train - watch pred: {watch_count}, share pred: {share_count}.')


def test(model, dataloader):
    watch_accs, share_accs = list(), list()
    watch_losses, share_losses = list(), list()
    watch_count, share_count = Counter(), Counter()
    srart_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            pred_y1 = model.get_watch(x)
            pred_y2 = model.get_share(x)
            y['watch'] = y['watch'].to(device)
            y['share'] = y['share'].to(device)
            watch_losses.append(model.watch_loss(pred_y1, y['watch']).item())
            share_losses.append(model.share_loss(pred_y2, y['share']).item())
            watch_accs.append(acc(pred_y1.argmax(dim=1), y['watch']))
            share_accs.append(acc(pred_y2.argmax(dim=1), y['share']))
            watch_count.update(pred_y1.argmax(dim=1).cpu().numpy())
            share_count.update(pred_y2.argmax(dim=1).cpu().numpy())
    time_cost = time.time() - srart_time
    print(f'test  - total batch: {len(dataloader):5.0f}, time cost: {time_cost:5.0f}, '
          f'watch acc: {mean(watch_accs):.4f}, share acc: {mean(share_accs):.4f}, '
          f'watch loss: {mean(watch_losses):.4f}, share loss:{mean(share_losses):.4f}.')
    print(f'test  - watch pred: {dict(watch_count)}, share pred: {dict(share_count)}.')


def main():
    os.system(f"rm -rf {dnn['savedir']} 2>&1 > /dev/null")
    os.makedirs(dnn['savedir'], exist_ok=True)

    print('--- loading data ---')
    start_time = time.time()
    behav_data = pd.read_csv(dataset['training_behavior_file'], sep='\t',
                             usecols=['user_id', 'video_id', 'is_share', 'watch_label'],
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
    start_time = time.time()
    train_behav, test_behav = train_test_split(split, behav_data)
    train_data = BehaviorDataset(user_data, video_data, train_behav)
    test_data  = BehaviorDataset(user_data, video_data, test_behav)
    sampler = BalancedResampler(w_labels=train_behav.watch_label,
                                s_labels=train_behav.is_share,
                                **dataset['sampler'])
    train_loader = DataLoader(train_data, batch_size=dnn['batch'],
                              sampler=sampler, num_workers=25)
    test_loader  = DataLoader(test_data, batch_size=dnn['batch'],
                              shuffle=True, num_workers=25)
    print(f'total training batch: {len(train_loader)}, test batch: {len(test_loader)}.')
    print(f'time cost of building dataset: {time.time() - start_time:.2f}s.')

    model = DIGIXNet(lr=lr).to(device)
    print('\n--- model structure ---')
    print(model)
    print('\n--- loss wright ---')
    print(model.watch_loss.weight.view(-1))
    print(model.share_loss.weight.view(-1))
    
    print(f'\n--- start training at {time.ctime()} ---')
    for e in range(epoch):
        print(f'epoch {e+1}/{epoch}:')
        train(model, train_loader)
        test(model, test_loader)
        state = dict(
            model=model.cpu().state_dict(),
            optim=model.optimizer.cpu().state_dict()
        )
        model_file = os.path.join(dnn['savedir'], f'{e}.pth')
        torch.save(state, model_file)
    print(f'\n--- training completed at {time.ctime()} ---')


if __name__ == '__main__':
    main()