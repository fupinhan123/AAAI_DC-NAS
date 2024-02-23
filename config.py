#encoding=utf-8


def get_configs():
    paras = {
        'data_name': 'data_set_rgb_depth',
        #'data_name': 'data_set_rgb_ske',
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        'feature_statistics': ['a', 'a', 'a', 'a', 'a'],
        'fused_nb_feats': 32,
        'nb_view':8 ,
        'pop_size': 28,
        'nb_iters': 15,
        'result_save_dir': 'DC-NAS-True-2023812' + '-32-split-3-' + 'result',
        'gpu_list': [0,2,3,4,5,6,7],
        'epochs': 100,
        'batch_size':64,
        'patience': 10,
        # EF
        'is_remove':  True,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': False,
        'max_len':30,
        # data set information
        'image_size': {
            'w': 224, 'h': 224, 'c': 3},
        'classes': 83,    #60
        'split_data' :[2,4,6,8,10],
        'fusion_L' : 4,
        'fusion_C' : 512,
    }
    return paras