import argparse
import csv
import logging
import os

import numpy as np
import pandas as pd

from config import dataset
from data.dtype import *
from data.preprocessing import *
from data.builder import multi_builder


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s'
        )


def arg_parser():
    parser = argparse.ArgumentParser('video preference dataset builder')
    parser.add_argument('--mode', '-m', type=str, help='dataset building mode: {train/test}.')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, help='create n subprocesses.')
    return parser.parse_args()
    

def main():
    args  = arg_parser()
    config = dataset
    if args.mode == 'train':
        behav_col = ['user_id', 'video_id', 'is_share', 'watch_label']
        behav_file = config['training_behavior_file']
    elif args.mode == 'test':
        behav_col = ['user_id', 'video_id']
        behav_file = config['test_behavior_file']
    else:
        raise ValueError('invalid dataset building mode.')

    # process and combine features.
    behav_data = pd.read_csv(behav_file, sep='\t', 
                             usecols=behav_col, dtype=behav_dtype)
    user_data  = pd.read_csv(config['user_file'], sep='\t',
                             index_col='user_id', dtype=user_dtype)
    user_data  = user_process(user_data)
    video_data = pd.read_csv(config['video_file'], sep='\t', 
                             index_col='video_id', dtype=video_dtype,
                             quoting=csv.QUOTE_NONE)
    video_data = video_process(video_data, cutoff=config['tag_cutoff'])

    # combine and save dataset.
    os.makedirs(config['save_dir'], exist_ok=True)
    filename = os.path.join(config['save_dir'], f'{args.mode}.npz')
    x = multi_builder(
        n_jobs=args.n_jobs,
        user_df=user_data,
        video_df=video_data,
        user_idx=behav_data.user_id.values,
        video_idx=behav_data.video_id.values,
        )
    if args.mode == 'train':
        y1 = behav_data.watch_label.values
        y2 = behav_data.is_share.values
        np.savez_compressed(filename, x=x, watch=y1, share=y2)
    else:
        np.savez_compressed(filename, x=x)


if __name__  == '__main__':
    main()