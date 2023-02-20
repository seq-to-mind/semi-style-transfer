from utils import get_available_gpu

use_cuda = True
gpu_id = str(get_available_gpu())

pretrained_model = "facebook/bart-base"
pretrained_tokenizer = "facebook/bart-base"
pretrained_style_model = "roberta-base"

corpus_mode = "Yelp"
# corpus_mode = "amazon"
# corpus_mode = "GYAFC"

# pseudo_method = "semantic"
pseudo_method = "lexical"

sentence_seg_token = " </s> <s> "
# sentence_seg_token = " [SEP] [CLS] "

using_label_smoothing = True
smooth_epsilon = 0.15

start_from_epoch = 0
supervised_epoch_num = 2

pure_unsupervised_training = False
MLE_teacher_forcing_anneal_rate = 1.0

train = True
save_model = False

load_model_path = "./saved_models/xxxx.pth"

batch_loss_print_interval = 20
print_all_predictions = True

batch_size = 64
num_epochs = 10

if corpus_mode in ["Yelp", "amazon"]:
    diversity_ctrl = True
    cyclic_balance = True
else:
    diversity_ctrl = False
    cyclic_balance = False
