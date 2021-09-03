__all__ = ['user_dtype', 'video_dtype', 'behav_dtype']

user_dtype = dict(
    age='int',
    gender='int',
    country='int',
    province='int',
    city='int',
    city_level='int',
    device_name='int'
)

video_dtype = dict(
    video_name='object',
    video_tags='object',
    video_description='object',
    video_release_date='object',
    video_director_list='object',
    video_actor_list='object',
    video_score='float',
    video_second_class='object',
    video_duration='int'
)

behav_dtype = dict(
    user_id='int',
    video_id='int',
    is_watch='category',
    is_share='int',
    is_collect='category',
    is_comment='category',
    watch_start_time='object',
    watch_label='int',
    pt_d='object'
)