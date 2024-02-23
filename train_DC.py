# encoding=utf-8
import os
from warnings import simplefilter

import gen_offspring_tree
import gen_offspring_tree_DC_K
import tree_to_strlist

simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import tensorflow as tf
# https://stackoverflow.com/questions/60130622/warningtensorflow-with-constraint-is-deprecated-and-will-be-removed-in-a-future
tf.get_logger().setLevel('ERROR')
import numpy as np
from os.path import join as pjoin
import random
from multiprocessing import Pool
import time
from  best_fives  import best_fivess,Archive1,Archive2
from  sklearn import  metrics
import config
import population_init
import utils
from code2net_tree import code2net_tree
from  datasetsplit import  split

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']
batch_size = paras['batch_size']
epochs = paras['epochs']
pop_size = paras['pop_size']
nb_iters = paras['nb_iters']
data_name = paras['data_name']
split_data = paras['split_data']


data_base_dir = os.path.join('', data_name)
data_lists = split.load_data_features()
dict_data ={'a':0,'b':1,'c':2,'d':3,'e':4}
def train_individual(individual_code, result_save_dir='.', gpu='0', iter_pop = 0,is_exist = False,pop_id = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    data_list = split.get_split_data(data_lists,pop_id)
    view_train_xx, view_test_xx = [], []
    sta_data_list = []
    views = []

    for value in individual_code:
        if(value[1] >= 'a' and  value[1] <= 'z'):
            sta_data_list.append(dict_data[value[1]])
            views.append(int(value[0]))

    for index,num in enumerate(sta_data_list):
        if num >= 5:
            num = 2
        data = data_list[num]
        view_train_xx.append(data[0][views[index]])
        view_test_xx.append(data[2][views[index]])
    view_train_x1, train_y, view_test_x1, test_y = data_list[0]


    individual_code_tree, nb_view = tree_to_strlist.viewfusion(individual_code)


    individual_code_str = '+'.join([str(ind) for ind in individual_code])



    nb_feats = [i.shape[1] for i in view_train_xx]


    checkpoint_filepath = os.path.join(result_save_dir, individual_code_str + '.h5')
    if is_exist == False:
      model = code2net_tree(individual_code=individual_code_tree,nb_feats=nb_feats,listtree = individual_code)
    else :
      model = tf.keras.models.load_model(checkpoint_filepath)
    adam = tf.keras.optimizers.Adam()
    topk = tf.keras.metrics.top_k_categorical_accuracy
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', topk])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=paras['patience'])
    csv_filepath = os.path.join(result_save_dir, individual_code_str + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)
    model.fit(view_train_xx, train_y, batch_size=batch_size, epochs=epochs,
              verbose=0, validation_data=(view_test_xx, test_y),
              callbacks=[csv_logger, early_stop, checkpoint])

    model_best = tf.keras.models.load_model(checkpoint_filepath)
    #sumper = model_best.summary()
    pre_y = model_best.predict(view_test_xx)
    pre_y = np.argmax(pre_y, axis=1)
    true_y = np.argmax(test_y, axis=1)
    acc = metrics.accuracy_score(true_y, pre_y)
    total_params = model_best.count_params()
    return individual_code_str + ',' + str(acc) + ','  + str(total_params)

def find_same_code_acc(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str

def record_code(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str


def list2str(list1):
    return '-'.join([str(i) for i in list1])

def list2str_tree(list1):
    return '+'.join([str(i) for i in list1])

def multi_proccess_train(i_iter, Q_t, shared_code_sets):


    gpu_list = paras['gpu_list']
    gpus = len(gpu_list)
    gpu_idx = 0
    pool = Pool(gpus)
    individual_code_str = []
    pop_size1 = len(Q_t)
    for ind_i in np.arange(0, pop_size1):
        if ind_i >= 0 and ind_i <= 6:
            pop_num = 3
        elif ind_i >= 7 and ind_i <= 13:
            pop_num = 3
        elif ind_i >= 14 and ind_i <= 20:
            pop_num = 3
        else:
            pop_num = 3
        print(len(Q_t), '==========', ind_i+1)
        code_str = list2str_tree(Q_t[ind_i])
        utils.write_result_file(','.join([str(i_iter+1), code_str]),fn=os.path.join(result_save_dir, 'history.csv'))
        if code_str not in shared_code_sets or code_str in best_fivess:
            if code_str not in shared_code_sets:
                is_exist = False
                shared_code_sets.add(code_str)
                individual_code_str.append(pool.apply_async(func=train_individual,args=(Q_t[ind_i], result_save_dir, str(gpu_list[gpu_idx]),i_iter,is_exist,pop_num)))
            gpu_idx += 1
        if gpu_idx == gpus or ind_i == (pop_size1-1):
            pool.close()
            pool.join()
            for ss in individual_code_str:
                utils.write_result_file(ss.get(), fn=os.path.join(result_save_dir, 'result.csv'))
            pool = Pool(gpus)
            gpu_idx = 0
            individual_code_str = []


def train():
    shared_code_sets = set()
    # 1. population initialization
    print(f'The number of views: {len(data_lists[0][0][0])}')
    ini_population = population_init.generate_population_tree(views=len(data_lists[0][0][0]), pop_size=pop_size, verbose=0)
    random.shuffle(ini_population)
    start = time.time()
    P_t = ini_population
    multi_proccess_train(i_iter=-1, Q_t=P_t, shared_code_sets=shared_code_sets)

    # 3. gen_offspring
    for i in tqdm(range(paras['nb_iters'])):
        print(f'==================={i+1}/', paras['nb_iters'])
        Q_t = gen_offspring_tree.gen_offspring(P_t)
        multi_proccess_train(i_iter=i, Q_t=Q_t, shared_code_sets=shared_code_sets)
        P_t = gen_offspring_tree.selection(P_t, Q_t)
        print('=' * 60, i+1, 'End.')

        print(f'Total time is :{time.time()-start}')
        utils.write_result_file(str(time.time()-start), fn=os.path.join(result_save_dir, 'history.csv'))


def train_one_code(is_train=False, code='8-9-1-4-7-2-5-1-0-0-0-4-4'):
    codes = code.split('+')
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    if  is_train == True:
        train_individual(codes, result_save_dir,'2',is_exist= True,pop_id = 4)





if __name__ == '__main__':
    result_save_dir = pjoin(data_name+'_view_result', paras['result_save_dir'])
    print(result_save_dir)
    print(data_name, fused_nb_feats)
    is_trian = True
    os.makedirs(result_save_dir, exist_ok=True)
    is_train_one_code = True
    if is_train_one_code == True:
        train_one_code(is_train = True,code='1a+6a+2a+-0+-0')
    else :
        train()
    print(result_save_dir)
    print(data_name, fused_nb_feats)