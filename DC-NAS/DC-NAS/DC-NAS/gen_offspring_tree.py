import random
import copy
import utils_tree
from config import get_configs
paras = get_configs()
nb_fusion_way = len(paras['fusion_ways'])
nb_view = 5  # paras['nb_view']
is_remove = paras['is_remove']
import  utils
from  best_fives import  best_fivess,Archive1,Archive2,A1_best_fivess,A2_best_fivess,A3_best_fivess,A4_best_fivess

def get_all_nodes_identifier(tree):
    nodes = tree.all_nodes()
    identifiersfph = []
    for node in nodes[:]:
        if tree.parent(node.identifier) is not None:
            identifiersfph.append(node.identifier)
        else:
            nodes.remove(node)
    return nodes, identifiersfph


def get_branch_nodes_identifier(tree):
    all_nodes = tree.all_nodes()
    branch_nodes = []
    identifiersfph = []
    for node in all_nodes[:]:
        if len(tree.is_branch(node.identifier)) != 0  and tree.parent(node.identifier) is not None:
            branch_nodes.append(node)
            identifiersfph.append(node.identifier)
        else:
            all_nodes.remove(node)
    return branch_nodes,identifiersfph


def get_leaf_nodes_identifier(tree):
    nodes = tree.leaves()
    identifiers = [node.identifier for node in nodes]
    return nodes, identifiers


def split_tree(tree, nid):
    tree_copy = copy.deepcopy(tree)
    removed_tree = tree_copy.remove_subtree(nid=nid, identifier=nid)
    return tree_copy, removed_tree


def crossover(tree1, tree2, crossover_rate, is_remove = is_remove,max_deep = 15):

    if len(tree1) ==0 or  len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:
        return tree1,tree2

    r = random.random()
    if(r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_all_nodes_identifier(tree1)
        tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)
        if len(tree1) == 1:
            print("!!!")
        tree1_split_point = random.choice(tree1_nodes)
        tree2_split_point = random.choice(tree2_nodes)
        

        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if(node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else :
            tree1_split_node_parent= node.identifier
        node = tree2.parent(tree2_split_node.identifier)
        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        tree1_left, tree1_right = split_tree(tree1, tree1_split_point.identifier)
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)
        tree1_left.paste(tree1_split_node_parent, tree2_right)
        tree2_left.paste(tree2_split_node_parent, tree1_right)
        if is_remove == True:
            tree1_left = quchong(tree1_left)
            tree2_left = quchong(tree2_left)
        if tree1_left.depth() > max_deep:
            tree1_left = quchong(tree1_left)
        if tree2_left.depth() > max_deep:
            tree2_left = quchong(tree2_left)

        return tree1_left,tree2_left
    else :
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        return tree1,tree2

def mutation(tree, mutation_rate, is_remove=is_remove, max_deep = 15):
    nodes = tree.all_nodes()

    node = random.choice(nodes)
    idtag = node.tag
    r = random.random()
    if (r < mutation_rate):
        if(idtag[0] == '-'):
            mutation_view = list(range(nb_fusion_way))
            id = random.choice(mutation_view)
            idtag = '-'+ str(id)
        else:
            mutation_view = list(range(nb_view))
            id = random.choice(mutation_view)
            idtag = str(id)
        node.tag = idtag
        if is_remove:
            tree = quchong(tree)
    else:
        if is_remove:
            tree = quchong(tree)

    if(tree.depth() > max_deep):
        tree = quchong(tree)
    return tree
def mutation_new_tree_crossover(tree, mutation_rate, is_remove=is_remove, max_deep = 15):
    r = random.random()
    if (r < mutation_rate):
        tree_mut = utils_tree.new_tree()
        tree1,tree2 = crossover(tree,tree_mut,1,is_remove,15)
        if is_remove:
            tree1 = quchong(tree1)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        return tree1
    else :
        if is_remove:
            tree = quchong(tree)
        if tree.depth() > max_deep:
            tree = quchong(tree)
        return tree

def quchong(tree_p):
    list_tree = utils_tree.tree_to_list2(tree_p)
    quchong_tree = []
    num_views = 0
    for i in list_tree:
        if i[0] != '-':
            if i not in quchong_tree:
               quchong_tree.append(i)
               num_views +=1
        else:
            if num_views >=2:
                quchong_tree.append(i)
                num_views-=1
    quchongtree = utils_tree.list_to_tree(quchong_tree)
    return quchongtree


def gen_offspring(P_t):
    cnt = 0
    shared_code_acc = utils.load_result()
    best_fivess.clear()
    Archive1.clear()
    Archive2.clear()
    for id, pop in enumerate(P_t):
        a = '+'.join([str(i) for i in pop])
        if id >= 0 and id <= 6:
             A1_best_fivess[a] = shared_code_acc[a]
        elif id >= 7 and id <= 13:
             A2_best_fivess[a] = shared_code_acc[a]
        elif id >= 14 and id <= 20:
            A3_best_fivess[a] = shared_code_acc[a]
        elif id >= 21  and id <= 27:
           A4_best_fivess[a] = shared_code_acc[a]

    sorted_dict1 = {k: v for k, v in sorted(A1_best_fivess.items(), key=lambda item: item[1], reverse=True)}
    sorted_dict2 = {k: v for k, v in sorted(A2_best_fivess.items(), key=lambda item: item[1], reverse=True)}
    sorted_dict3 = {k: v for k, v in sorted(A3_best_fivess.items(), key=lambda item: item[1], reverse=True)}
    sorted_dict4 = {k: v for k, v in sorted(A4_best_fivess.items(), key=lambda item: item[1], reverse=True)}

    A_1_top_three_keys = list(sorted_dict1.keys())[:3]
    A_1_top_three_keys +=list(sorted_dict2.keys())[:2]
    A_1_top_three_keys +=list(sorted_dict3.keys())[:2]

    A_2_top_three_keys = list(sorted_dict4.keys())[:7]

    for pop in A_1_top_three_keys:
        Archive1.append(pop.split('+'))

    for pop in A_2_top_three_keys:
        Archive2.append(pop.split('+'))

    def select_p(id = 0):
        two = []
        if id >= 0 and id <= 6:
            two = random.sample(range(0,7), 2)
        elif id >= 7 and id <= 13:
            two = random.sample(range(7, 14), 2)
        elif id >= 14 and id <= 20:
            two = random.sample(range(14, 21), 2)
        elif id >= 21 and id <= 27:
            two = random.sample(range(21, 28), 2)
        a1 = '+'.join([str(i) for i in P_t[two[0]]])
        a2 = '+'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]
        return p1
    def select_p_1(id = 0):
        if id >= 0 and id <= 20:
            another_one = random.sample(range(len(Archive2)),1)
            p2 = Archive2[another_one[0]]
            return p2
        elif id >= 21 and id <= 27:
            another_one = random.sample(range(len(Archive1)),1)
            p2 = Archive1[another_one[0]]
            return p2
    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p(len(Q_t))
        p2 = select_p_1(len(Q_t))
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):  ## 防止重复
            p1 = select_p(len(Q_t))
            p2 = select_p_1(len(Q_t))
            cnt += 1
            if cnt == 3:
                break
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)
        o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
        o1 = utils_tree.tree_to_list2(o1_tree)
        o2 = utils_tree.tree_to_list2(o2_tree)
        Q_t.append(o1)
        Q_t.append(o2)
    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)
        p1_tree = mutation_new_tree_crossover(p_tree,mutation_rate=paras['mutation_rate'])
        p1 = utils_tree.tree_to_list2(p1_tree)
        Q_tt.append(p1)
    Q_t = Q_tt

    return Q_t

def selection(P_t, Q_t):
    shared_code_acc = utils.load_result()
    # print(f'P_t: {P_t}')
    # print(f'Q_t: {Q_t}')
    # print(f'f: {shared_code_acc}')
    def select_p1(select_pool1,select_pool2,id):
        one = 0
        two = 0
        if id >= 0 and id <= 6:
            one = random.sample(range(0, 7), 1)
            two = random.sample(range(0, 7), 1)
        elif id >= 7 and id <= 13:
            one = random.sample(range(7, 14), 1)
            two = random.sample(range(7, 14), 1)
        elif id >= 14 and id <= 20:
            one = random.sample(range(14, 21), 1)
            two = random.sample(range(14, 21), 1)
        elif id >= 21 and id <= 27:
            one = random.sample(range(21, 28), 1)
            two = random.sample(range(21, 28), 1)
        a1 = '+'.join([str(i) for i in select_pool1[one[0]]])
        a2 = '+'.join([str(i) for i in select_pool2[two[0]]])
        p1 = select_pool1[one[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool2[two[0]]
        return p1
    P_t1 = []
    #Pt_Qt = P_t+Q_t
    while len(P_t1) < len(P_t):
        p = select_p1(P_t,Q_t,len(P_t1))
        P_t1.append(p)

    max_code = []
    max_code_str = ''
    min_code_str =''
    for k, v in shared_code_acc.items():
        if v == max(shared_code_acc.values()):
            max_code_str = k
            max_code = k.strip().split('+')
        if v == min(shared_code_acc.values()):
            min_code_str = k
    is_max = False
    for i, v in enumerate(P_t1):
        v_str = utils_tree.tree_list2str(v)
        if v_str == max_code_str:
            is_max = True
            break
    if not is_max:
        min_i = 0
        for i, v in enumerate(P_t1):
            v_str = utils_tree.tree_list2str(v)
            if v_str == min_code_str:
                min_i = i
                break
        P_t1[min_i] = max_code
    return P_t1



