import numpy as np
from sklearn.model_selection import train_test_split
from  data_utils import  data_uitl
import  os
import config
from sklearn.model_selection import StratifiedShuffleSplit
paras = config.get_configs()
data_name = paras['data_name']

data_base_dir = os.path.join('/export/fupinhan/datasetENAS/', data_name)

view_data_dir1 = os.path.join(data_base_dir, 'view')
view_data_dir2 = os.path.join(data_base_dir, 'view1')
view_data_dir3 = os.path.join(data_base_dir, 'view2')
view_data_dir4 = os.path.join(data_base_dir, 'view3')
view_data_dir5 = os.path.join(data_base_dir, 'view4')


def split_train_test(x, y, n_splits=3, test_size=0.7, seed=1024):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(x, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    return train_idxs, test_idxs


def split_data(view_train_x, train_y, split_percentages):
    splits = []
    split_data_y = []
    for split_percentage in split_percentages:
        split_data = []
        y_split = []
        for i,modality_data in enumerate(view_train_x):
            X_split, _, y_split, _ = train_test_split(modality_data, train_y, train_size=(split_percentage)/100, random_state=42,shuffle=False)
            split_data.append(X_split)
        splits.append(split_data)
        split_data_y.append(y_split)

    return splits,split_data_y


def splits_data(view_train_x, train_y, split_percentages):
    splits = []
    split_data_y = []
    train_idxs, test_idxs = split_train_test(x=view_train_x[0], y=train_y, n_splits=3,test_size=0.3, seed=1024)
    for split_percentage in split_percentages:
        split_data = []
        for i,modality_data in enumerate(view_train_x):
            split_data.append(modality_data[train_idxs[split_percentage]])
        splits.append(split_data)
        split_data_y.append(train_y[train_idxs[split_percentage]])

    return splits,split_data_y

def load_data_features():
    view_train_x1, train_y1, view_test_x1, test_y1 = data_uitl.get_views(view_data_dir=view_data_dir1)
    view_train_x2, train_y2, view_test_x2, test_y2 = data_uitl.get_views(view_data_dir=view_data_dir2)
    view_train_x3, train_y3, view_test_x3, test_y3 = data_uitl.get_views(view_data_dir=view_data_dir3)
    view_train_x4, train_y4, view_test_x4, test_y4 = data_uitl.get_views(view_data_dir=view_data_dir4)
    view_train_x5, train_y5, view_test_x5, test_y5 = data_uitl.get_views(view_data_dir=view_data_dir5)
    split_percentages = [0,1,2]

    splits_1,split_y_1 = splits_data(view_train_x1, train_y1, split_percentages)
    splits_2,split_y_2 = splits_data(view_train_x2, train_y2, split_percentages)
    splits_3,split_y_3 = splits_data(view_train_x3, train_y3, split_percentages)
    splits_4,split_y_4 = splits_data(view_train_x4, train_y4, split_percentages)
    splits_5,split_y_5 = splits_data(view_train_x5, train_y5, split_percentages)


    splits_1.append(view_train_x1)
    split_y_1.append(train_y1)

    splits_2.append(view_train_x2)
    split_y_2.append(train_y2)

    splits_3.append(view_train_x3)
    split_y_3.append(train_y3)

    splits_4.append(view_train_x4)
    split_y_4.append(train_y4)

    splits_5.append(view_train_x5)
    split_y_5.append(train_y5)

    split_percentages.append(3)
    data_list = [
        [splits_1, split_y_1, view_test_x1, test_y1],
        [splits_2, split_y_2, view_test_x2, test_y2],
        [splits_3, split_y_3, view_test_x3, test_y3],
        [splits_4, split_y_4, view_test_x4, test_y4],
        [splits_5, split_y_5, view_test_x5, test_y5]
    ]
    return data_list


def get_split_data(data_list,iter_pop = 0):
    data_list_split = [
        [data_list[0][0][iter_pop], data_list[0][1][iter_pop], data_list[0][2], data_list[0][3]],
        [data_list[1][0][iter_pop], data_list[1][1][iter_pop], data_list[1][2], data_list[1][3]],
        [data_list[2][0][iter_pop], data_list[2][1][iter_pop], data_list[2][2], data_list[2][3]],
        [data_list[3][0][iter_pop], data_list[3][1][iter_pop], data_list[3][2], data_list[3][3]],
    ]
    return data_list_split






