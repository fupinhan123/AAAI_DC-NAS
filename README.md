# **DC-NAS: Divide-and-Conquer Neural Architecture Search for Multi-Modal Classification（AAAI 2024 Oral）**
**Description**: The software package includes Python code for the DC-NAS algorithm used for multi-modal feature fusion.
It can perform a search for multi-modal feature fusion methods and significantly improve algorithm efficiency using a divide-and-conquer approach.
This software package has been applied to multi-modal classification tasks.

**Requirement**: The package was developed with python3 and tensorflow-gpu(2.0.3).

**ATTN**: This package is free for academic usage. You can run it at your own risk.


Using their respective multimodal datasets only requires creating a file for the dataset, saving it, and placing it in the root directory.
In the code, we have provided a directory named "data_set_rgb_depth" for the dataset. You just need to place your dataset inside this directory.
We utilized the same hardware environment as the MFAS framework, which includes an NVIDIA Tesla P100 with 16 GB of GPU memory.


Before running train_DC-NAS.py, you have to set some parameters in ```config.py``` file.

```python
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
```

```python
    $python train_DC.py
```
