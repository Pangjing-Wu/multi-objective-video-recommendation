import logging
import time
from multiprocessing import Process, Queue

import numpy as np


__all__ = ['multi_builder']


def multi_builder(n_jobs, user_df, video_df,
                   user_idx, video_idx):
    assert n_jobs > 0
    assert len(user_idx) == len(video_idx)
    # split task
    user_idxs  = np.array_split(user_idx, n_jobs)
    video_idxs = np.array_split(video_idx, n_jobs)
    # create multiprocess pool.
    process_pool = list()
    queue = Queue(n_jobs)
    for i in range(n_jobs):
        kwargs = dict(
            queue=queue,
            user_df=user_df,
            video_df=video_df,
            user_idx=user_idxs[i],
            video_idx=video_idxs[i],
            )
        process_pool.append(
            Process(
                target=__data_combine,
                name=f'Combiner {i}',
                kwargs=kwargs
            )
        )
    # start subprocess.
    for process in process_pool:
        process.start()
        logging.info(f'start subprocess {process.name}.')
        time.sleep(1)
    # join subprocess until terminated.
    for process in process_pool:
        process.join()
        logging.info(f'[INFO] subprocess {process.name} terminate.')
        time.sleep(1)
    # combine data
    arr = list()
    while not queue.empty():
        arr.append(queue.get())
    return np.array(arr)


def __data_combine(queue, user_df, video_df,
                   user_idx, video_idx) -> np.array:
    arr = list()
    for i, j in zip(user_idx, video_idx):
        arr.append(
            np.hstack(
                (user_df.loc[i].values,
                 video_df.loc[j].values)
            )
        )
    queue.put(np.array(arr))
    return 0