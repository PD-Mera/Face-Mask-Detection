import random

paths_txt_file = './data/paths.txt'
train_paths_txt_file = './data/train_paths.txt'
valid_paths_txt_file = './data/valid_paths.txt'
ratio = "9:1"

with open(paths_txt_file, mode = "r") as f:
    data = f.readlines()

random.shuffle(data)
number_data = len(data)
train_ratio = float(ratio.split(":")[0]) / (float(ratio.split(":")[0]) + float(ratio.split(":")[1]))

train_data = data[:int(train_ratio * number_data)]
valid_data = data[int(train_ratio * number_data):]

with open(train_paths_txt_file, mode = "w") as f:
    f.writelines(train_data)

with open(valid_paths_txt_file, mode = "w") as f:
    f.writelines(valid_data)