import torch 
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from utils import get_batches
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from Model import StyleClassifier

# import models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")

model = model.cuda().eval()

text_file_0 = "amazon_data/sentiment.train.0"
text_file_1 = "amazon_data/sentiment.train.1"

text_list_0 = open(text_file_0, encoding="utf-8").readlines()
text_list_1 = open(text_file_1, encoding="utf-8").readlines()

text_list_0 = [i.strip() for i in text_list_0 if len(i.split()) > 5]
text_list_1 = [i.strip() for i in text_list_1 if len(i.split()) > 5]

text_list_0 = list(set(text_list_0))
text_list_1 = list(set(text_list_1))

print("text_list_0", len(text_list_0))
print("text_list_1", len(text_list_1))

style_classifier = StyleClassifier().cuda().eval()


def polarity_filter_out(text_list, classifier_model, class_type):
    tmp_batches = get_batches(text_list, batch_size=256)
    tmp_output_list = []
    for one_batch in tqdm(tmp_batches):
        style_CLS_logits, _ = classifier_model.binary_cls(["# " + i for i in one_batch])
        probs = F.softmax(style_CLS_logits, dim=1).cpu().numpy().tolist()
        tmp_output_list.extend([one_batch[k] for k, v in enumerate(probs) if v[class_type] > 0.7])
    return tmp_output_list


with torch.no_grad():
    text_list_0 = polarity_filter_out(text_list_0, style_classifier, class_type=0)
    text_list_1 = polarity_filter_out(text_list_1, style_classifier, class_type=1)

print("text_list_0", len(text_list_0))
print("text_list_1", len(text_list_1))


def get_all_vector_list(text_list):
    tmp_tensor = None
    tmp_batches = get_batches(text_list, batch_size=256)
    for one_batch in tqdm(tmp_batches):
        batch_input = tokenizer(one_batch, padding=True, truncation=True, return_tensors="pt")
        tmp_hidden = model(batch_input['input_ids'].cuda(), attention_mask=batch_input['attention_mask'].cuda(), output_hidden_states=False, return_dict=True).pooler_output.detach()
        if tmp_tensor is None:
            tmp_tensor = tmp_hidden.detach()
        else:
            tmp_tensor = torch.cat([tmp_tensor, tmp_hidden], dim=0).detach()
    return tmp_tensor

with torch.no_grad():
    tmp_tensor_0 = get_all_vector_list(text_list_0).detach()
    tmp_tensor_1 = get_all_vector_list(text_list_1).detach()

sentence_list_A = []
sentence_list_B = []

for i in tqdm(range(len(text_list_0))):
    similar_scores = F.cosine_similarity(tmp_tensor_0[i].unsqueeze(0).expand(tmp_tensor_1.size(0), -1), tmp_tensor_1, dim=1)
    one_best = torch.argmax(similar_scores).item()
    # print(similar_scores[one_best])
    # print(text_list_0[i], text_list_1[one_best])
    if similar_scores[one_best] > 0.85:
        sentence_list_A.append(text_list_0[i]+"\n")
        sentence_list_B.append(text_list_1[one_best]+"\n")

open("merged_A", "w").writelines(sentence_list_A)
open("merged_B", "w").writelines(sentence_list_B)

