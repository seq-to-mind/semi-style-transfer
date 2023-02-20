import json
import re
import numpy as np
import os


def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])

    return batches


def get_available_gpu():
    res = os.popen("nvidia-smi").readlines()

    gpu_list = []
    gpu_id = 0
    for i in res:
        memory_usage = re.findall("(\d+)MiB /", i)
        if len(memory_usage) > 0:
            gpu_list.append((gpu_id, int(memory_usage[0])))
            gpu_id += 1
    if len(gpu_list) > 1:
        gpu_list = gpu_list[1:]
    gpu_list = sorted(gpu_list, key=lambda k: k[1], reverse=False)
    print("Current GPU usage:", gpu_list, "\nGet the GPU with lowest usage:", gpu_list[0][0])
    if gpu_list[0][1] > 10000:
        print("\n#ERORR# There is no idle gpu.")
        exit()
    else:
        return gpu_list[0][0]
