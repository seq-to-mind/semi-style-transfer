import os
import numpy as np
import re
import logging
from nltk.stem import PorterStemmer
from difflib import SequenceMatcher
import editdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import random
from multiprocessing import Process, Queue

from utils import get_batches

""" Match Type Definition """
match_type_names = {1: "semantic_match", 2: "least_edit_distance", 3: "tf_idf"}

match_type = 2

print("The match method is", match_type_names[match_type])

# sentence similarity start here onwards
print("\nStarting sentence similarity\n")


A_path = "Yelp_data/sentiment.train.0"
B_path = "Yelp_data/sentiment.train.1"


def generate_content_list_job(input_split, task_id, que):
    que.put(find_best_match(input_split, task_id))


def multi_threading_crawler(input_list):
    q = Queue()
    threads = []
    print("Multi_thread size:", len(input_list))
    for i in range(len(input_list)):
        t = Process(target=generate_content_list_job, args=(input_list[i], i, q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
        # print("finish")
    results = []
    for _ in range(len(input_list)):
        results.append(q.get())
    return results


def find_best_match(id_split, task_id):
    style_A_list = open(A_path, encoding="utf-8").readlines()
    style_B_list = open(B_path, encoding="utf-8").readlines()

    file_A_fp = open("./tmp/" + str(task_id) + "_A", 'w')
    file_B_fp = open("./tmp/" + str(task_id) + "_B", 'w')

    style_A_list_after_stemming = [" ".join([j for j in i.split() if len(j) > 3]) for i in style_A_list]
    style_B_list_after_stemming = [" ".join([j for j in i.split() if len(j) > 3]) for i in style_B_list]

    for tmp_i in tqdm(range(len(id_split)), leave=False):
        top_k_indices = []

        A_one_sen = style_A_list_after_stemming[id_split[tmp_i]]
        A_one_sen_raw_id = id_split[tmp_i]
        score_list = []

        for i in range(len(style_B_list_after_stemming)):
            if abs(len(A_one_sen) - len(style_B_list_after_stemming[i])) < 15:
                score_list.append(editdistance.distance(A_one_sen, style_B_list_after_stemming[i]))
            else:
                score_list.append(99999)

        top_k_indices = np.argsort(score_list)[:3]

        file_A_fp.write(style_A_list[A_one_sen_raw_id])
        file_B_fp.write(style_B_list[top_k_indices[0]])


input_batches = get_batches(list(range(len(open(A_path, encoding="utf-8").readlines()))), batch_size=10000)

multi_threading_crawler(input_batches)


""" merge the multi_threading output files """
sentence_list_A = []
sentence_list_B = []

for i in range(18):
    sentence_list_A.extend(open("./tmp/"+str(i) + "_A").readlines())
    sentence_list_B.extend(open("./tmp/"+str(i) + "_B").readlines())

assert len(sentence_list_A) == len(sentence_list_B)

print(len(sentence_list_A), len(sentence_list_B))
print(len(set(sentence_list_A)), len(set(sentence_list_B)))

open("merged_A", "w").writelines(sentence_list_A)
open("merged_B", "w").writelines(sentence_list_B)

