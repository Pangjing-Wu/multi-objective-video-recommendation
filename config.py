dataset = dict(
    user_file='./datasets/raw/traindata/user_features_data.csv',
    video_file='./datasets/raw/traindata/video_features_data.csv',
    # training_behavior_file='./datasets/raw/traindata/history_behavior_light.csv',
    training_behavior_file='./datasets/raw/traindata/history_behavior_data.csv',
    test_behavior_file='./datasets/raw/testdata/test.csv',
    save_dir='./datasets/array',
    video=dict(
        fill_nan=dict(
            int=0,
            str='<nan>',
            date='1850-01-01'
        ),
        bins=dict(
            score=10,
            duration=20,
            year=15
        ),
        tags_kwargs=dict(
            length=8,
            pad_term='<nan>'
            ),
        second_class_kwargs=dict(
            length=5,
            pad_term='<nan>'
            )
    ),
    user_col=['age', 'gender', 'country', 'province',
              'city', 'city_level', 'device_name'],
    video_col=['video_release_date', 'video_score',
               'video_second_class', 'video_duration'],
    sampler=dict(
        n_per_class=1000000,
        sampler_cache='./.cache/sampler.npy'
    )
)

dnn = dict(
    savedir='./results/dnn',
    batch=2048,
    user=dict(
        feature_name=dataset['user_col'],
        input_dim=[8, 4, 3, 33, 339, 8, 1826],
        em_dim=[4, 2, 2, 6, 10, 4, 72]
    ),
    afm=dict(
        field_dims=[9, 5, 4, 34, 340, 9,
                   1827, 10, 20, 15, 0,
                   0, 0, 0, 145],
        embed_dim=256,
        output_dim=128,
        atten_hidden_dim=256,
        dropout=0.3
    ),
    mtmlp=dict(
        input_dim=128,
        output_dim=[10, 2],
        public_hidden_dim=[256,],
        id_hidden_dim=[256,],
        dropout_rate=0.3
    ),
    watch_loss=dict(
        n=10,
        weight=None,
        gamma=2,
        ),
    share_loss=dict(
        n=2,
        weight=[1., 60.],
        gamma=3,
    )
)