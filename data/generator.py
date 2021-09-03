import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler


np.random.seed(0)


def train_test_split(ratio:float, behav_df):
    n = int(ratio * behav_df.shape[0])
    train_set = behav_df.iloc[:n]
    test_set  = behav_df.iloc[n:]
    return (train_set, test_set)


class BehaviorDataset(Dataset):

    def __init__(self, user_df, video_df, behav_df, inculde_label=True):
        self.user_df  = user_df
        self.video_df = video_df
        self.behav_df = behav_df
        self.inculde_label = inculde_label

    def __len__(self):
        return self.behav_df.shape[0]

    def __getitem__(self, index):
        i_behav = self.behav_df.iloc[index]
        user    = self.user_df.loc[i_behav.user_id]
        video   = self.video_df.loc[i_behav.video_id]
        x = np.hstack((user.values, video.values))
        if self.inculde_label:
            y = dict(
                watch=i_behav.watch_label,
                share=i_behav.is_share
            )
            return x, y
        else:
            return x


class BalancedResampler(Sampler):

    def __init__(self, n_per_class, w_labels,
                 s_labels, sampler_cache):
        self.n_per_class = n_per_class
        self.w_labels    = w_labels.reindex(range(len(w_labels)))
        self.s_labels    = s_labels
        self.n_class     = w_labels.value_counts().shape[0]
        self.sample_list = self.__load_cache(sampler_cache)
        if self.sample_list is None:
            self.sample_list = self.__build_sample_list(sampler_cache)

    def __iter__(self):
        return(iter(self.sample_list))

    def __len__(self):
        return int(self.n_per_class * self.n_class)

    def __load_cache(self, cache_file):
        if os.path.exists(cache_file):
            sample_list = np.load(cache_file)
            if len(sample_list) == len(self):
                print(f'load sample list from "{cache_file}".')
                return sample_list
            else:
                return None
        else:
            return None
    
    def __build_sample_list(self, cache_file):
        sample_list = list()
        for i in range(self.n_class):
            w_label = self.w_labels[self.w_labels == i]
            if len(w_label) < self.n_per_class:
                replace = True 
            else:
                replace = False
            sample1 = w_label[self.s_labels == 1].index.values.tolist()
            sample0 = w_label[self.s_labels == 0].sample(
                n=self.n_per_class - len(sample1),
                replace=replace,
                random_state=1
                ).index.values.tolist()
            sample_list += sample0 + sample1
        np.random.shuffle(sample_list)
        np.save(cache_file, sample_list)
        print(f'save sample list to "{cache_file}".')
        return sample_list