from  datasetsplit import  split

pop_id = 3
data_list = split.load_data_features(pop_id)
print(len(data_list))
print(len(data_list[0]))
print(len(data_list[0][0]))
print(len(data_list[0][0][7]))
print(len(data_list[0][1]))
