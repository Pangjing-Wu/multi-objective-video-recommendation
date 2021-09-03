import numpy as np
import pandas as pd

from .vocab import get_word_vector


__all__ = ['user_process', 'video_process']


max_min_norm = lambda x: (x - x.min()) / (x.max() - x.min())


def user_process(df:pd.DataFrame) -> pd.DataFrame:
    ''' preprocess users' features.
    '''
    df = df + 1
    # df = df.drop(['country', 'city'], axis=1)
    return df


def video_process(df:pd.DataFrame, fill_nan:dict, bins:dict,
                  tags_kwargs:dict, second_class_kwargs:dict) -> pd.DataFrame:
    ''' process videos' feature.
    1. drop ineffective columns;
    2. fill nan record;
    3. process non-numerical features;
    4. process numerical features;
    5. combine features.
    :return: np.array, [score, duration, release_year, second_class, tags]
    '''
    # drop ineffective columns
    df = df.drop(['video_name', 'video_description', 
                  'video_director_list', 'video_actor_list'], axis=1)
    
    # fill nan
    df.video_tags  = df.video_tags.fillna(fill_nan['str'])
    df.video_score = df.video_score.fillna(fill_nan['int'])
    df.video_release_date = df.video_release_date.fillna(fill_nan['date'])
    df.video_second_class = df.video_second_class.fillna(fill_nan['str'])
    
    # process non-numerical features
    ## date
    df.video_release_date = pd.to_datetime(df.video_release_date)
    release_year = df.video_release_date.apply(lambda x: x.year)
    release_year = max_min_norm(release_year.values)
    release_year = np.digitize(release_year, np.arange(0, 1, 1/bins['year']))
    ## class tag
    # tags = get_word_vector(
    #     df.video_second_class,
    #     **tags_kwargs
    #     )
    second_class = get_word_vector(
        df.video_second_class,
        **second_class_kwargs
        )
    # process numerical features
    score    = max_min_norm(df.video_score.values)
    duration = max_min_norm(df.video_duration.values)
    score    = np.digitize(score, np.arange(0, 1, 1/bins['score']))
    duration = np.digitize(duration, np.arange(0, 1, 1/bins['duration']))
    # combine features
    score        = score.reshape(-1, 1)
    duration     = duration.reshape(-1, 1)
    release_year = release_year.reshape(-1, 1)
    # data = np.hstack((score, duration, release_year, second_class, tags))
    data = np.hstack((score, duration, release_year, second_class))
    col  = ['score', 'duration', 'release_year']
    col += [f'second_class{i+1}' for i in range (second_class_kwargs['length'])]
    # col += [f'tags{i+1}' for i in range (tags_kwargs['length'])]
    return pd.DataFrame(data, columns=col, index=df.index)