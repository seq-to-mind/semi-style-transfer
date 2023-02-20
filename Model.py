import random
import torch
import torch.nn as nn

import numpy as np
import re
import global_config
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, RobertaForMaskedLM, RobertaTokenizer, RobertaForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss
    smooth_loss = smooth_loss
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss.mean(), nll_loss.mean()


class NeuralSeq2Seq(nn.Module):
    def __init__(self):
        super(NeuralSeq2Seq, self).__init__()

        self.language_backbone = BartForConditionalGeneration.from_pretrained(global_config.pretrained_model, output_hidden_states=False)
        self.tokenizer = BartTokenizer.from_pretrained(global_config.pretrained_tokenizer, use_fast=True)

        self.bleu_smooth_function = SmoothingFunction()

    def rewrite_sentence_supervised(self, batch_input_tensor, batch_attention_mask, batch_decoder_label_tensor, reduce_CE_dim=True):
        output_logits = self.language_backbone(input_ids=batch_input_tensor, attention_mask=batch_attention_mask, labels=batch_decoder_label_tensor).logits

        if global_config.using_label_smoothing:
            supervised_loss, _ = label_smoothed_nll_loss(lprobs=F.log_softmax(output_logits, dim=2), target=batch_decoder_label_tensor,
                                                         epsilon=global_config.smooth_epsilon, ignore_index=None)
        else:
            if reduce_CE_dim:
                supervised_loss = F.cross_entropy(output_logits.transpose(1, 2), batch_decoder_label_tensor)
            else:
                supervised_loss = torch.mean(F.cross_entropy(output_logits.transpose(1, 2), batch_decoder_label_tensor, reduction="none"), dim=1)

        return output_logits, supervised_loss

    def rewrite_sentence_RL(self, batch_input_tensor, batch_attention_mask, greedy_generation=True, plus_sampling_generation=False, set_min_length=7, set_max_length=25):
        output_greedy_sen_token_ids, output_greedy_sen_text, output_greedy_logp, output_greedy_one_hot = None, None, None, None
        if greedy_generation:
            output_greedy_sen_token_ids, output_greedy_sen_token_logits = \
                self.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask, output_scores=False, return_dict_in_generate=False,
                                                do_sample=False, num_beams=1, early_stopping=False, min_length=set_min_length, max_length=set_max_length, use_cache=False)
            """ remove the first decoding token </s> """
            output_greedy_sen_token_ids = output_greedy_sen_token_ids[:, 1:]
            output_greedy_sen_text = [self.tokenizer.decode(i, skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in output_greedy_sen_token_ids]
            output_greedy_logp = F.log_softmax(torch.cat(output_greedy_sen_token_logits, dim=1), dim=2)
            output_greedy_one_hot = F.one_hot(output_greedy_sen_token_ids, num_classes=self.tokenizer.vocab_size)

        output_sampling_sen_token_ids, output_sampling_sen_text, output_sampling_logp, output_sampling_one_hot = None, None, None, None
        if plus_sampling_generation:
            output_sampling_sen_token_ids, output_sampling_sen_token_logits = \
                self.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask, output_scores=False, return_dict_in_generate=False,
                                                do_sample=True, num_beams=1, early_stopping=False, min_length=set_min_length, max_length=set_max_length, use_cache=False)
            output_sampling_sen_token_ids = output_sampling_sen_token_ids[:, 1:]
            output_sampling_sen_text = [self.tokenizer.decode(i, skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in output_sampling_sen_token_ids]
            output_sampling_logp = F.log_softmax(torch.cat(output_sampling_sen_token_logits, dim=1), dim=2)
            output_sampling_one_hot = F.one_hot(output_sampling_sen_token_ids, num_classes=self.tokenizer.vocab_size)

        return output_greedy_sen_token_ids, output_greedy_sen_text, output_greedy_logp, output_greedy_one_hot, \
               output_sampling_sen_token_ids, output_sampling_sen_text, output_sampling_logp, output_sampling_one_hot

    def cyclic_generation(self, target_style_list, trans_sen_text_input, raw_text_input_with_label):
        """  write the sentence back for cyclic loss """
        trans_sen_text_input = [i.replace("<pad>", "") for i in trans_sen_text_input]
        back_to_input = self.tokenizer(trans_sen_text_input, return_tensors='pt', padding=True, add_special_tokens=False)
        back_to_input_tensor = back_to_input.data["input_ids"].cuda()
        back_to_attention_mask = back_to_input.data["attention_mask"].cuda()

        raw_encoded_label = self.tokenizer(raw_text_input_with_label, return_tensors='pt', padding=True, add_special_tokens=False)
        raw_encoded_label_ids = raw_encoded_label.data["input_ids"].cuda()
        raw_encoded_label_mask = raw_encoded_label.data["attention_mask"].cuda()

        """ cyclic back translation version 2.0 """
        with torch.no_grad():
            _, output_text, output_greedy_logp, _, _, _, _, _, = self.rewrite_sentence_RL(back_to_input_tensor, back_to_attention_mask, plus_sampling_generation=False)

        use_bleu_score_for_cyclic = True
        if use_bleu_score_for_cyclic:
            output_text = [i.replace("<pad>", "") for i in output_text]

            batch_bleu_score = [sentence_bleu([word_tokenize(raw_text_input_with_label[i])], word_tokenize(output_text[i]),
                                              weights=(0.15, 0.15, 0.35, 0.35), smoothing_function=self.bleu_smooth_function.method1) for i in range(len(output_text))]

            output_reward = torch.tensor(batch_bleu_score).cuda()
        else:
            min_length = min(output_greedy_logp.size(1), raw_encoded_label_ids.size(1))
            tmp_nll_loss = - output_greedy_logp[:, :min_length, :].gather(dim=-1, index=raw_encoded_label_ids[:, :min_length].unsqueeze(2))
            tmp_nll_loss = tmp_nll_loss.squeeze(2) * raw_encoded_label_mask[:, :min_length]
            tmp_nll_loss = torch.sum(tmp_nll_loss, dim=1) / torch.sum(raw_encoded_label_mask[:, :min_length], dim=1)
            output_reward = - tmp_nll_loss

        return output_reward


class StyleClassifier(nn.Module):
    def __init__(self):
        super(StyleClassifier, self).__init__()
        """ Roberta and BART are using the same tokenizer """
        print("Loading BERT model for style classification:", global_config.corpus_mode)
        self.style_classifier_tokenizer = BartTokenizer.from_pretrained(global_config.pretrained_tokenizer, use_fast=True)
        self.style_classifier.cuda().eval()
        self.style_classifier.load_state_dict(torch.load('saved_models/TextBERT_' + global_config.corpus_mode + '/TextBERT_best.chkpt'))

    def binary_cls(self, batch_text, batch_label=None):
        batch_text = [i.replace("</s>", "").replace("<s>", "") for i in batch_text]
        batch_text = [i[i.index("#") + 1:] if "#" in i else i for i in batch_text]
        if global_config.corpus_mode in ["amazon", "Yelp", "GYAFC"]:
            batch_input = self.style_classifier_tokenizer(batch_text, add_special_tokens=True, padding=True, return_tensors="pt").data
            logits = self.style_classifier(batch_input["input_ids"].cuda(), attention_mask=batch_input["attention_mask"].cuda()).logits.detach()
            prediction = torch.argmax(logits, dim=-1)
        else:
            batch_input = self.style_classifier_tokenizer(batch_text, add_special_tokens=False, padding=True, return_tensors="pt").data['input_ids'].cuda()
            logits = self.style_classifier(batch_input).detach()
            prediction = torch.argmax(logits, dim=-1)
        return logits, prediction


class Model:
    def __init__(self):
        self.agent = NeuralSeq2Seq()
        if global_config.use_cuda:
            self.agent.cuda()

        self.style_CLS = StyleClassifier()
        if global_config.use_cuda:
            self.style_CLS.cuda()

        self.RL_training = False
        self.iter_step = 1
        self.supervised_loss_decay = 1.0
        self.teacher_forcing_rate = 1.0

        self.diversity_pool = []
        self.diversity_dict = {}

        self.one_average_lambda = None

        self.style_class_dict = {"informal": 0, "formal": 1, "negative": 0, "positive": 1}

        if global_config.corpus_mode in ["amazon", "Yelp"]:
            self.optimizer_Supervised = torch.optim.RMSprop(params=self.agent.parameters(), lr=0.00002)
            self.optimizer_RL = torch.optim.RMSprop(params=self.agent.parameters(), lr=0.00002)

        elif global_config.corpus_mode == "GYAFC":
            self.optimizer_Supervised = torch.optim.AdamW(params=self.agent.parameters(), lr=0.00002)
            self.optimizer_RL = torch.optim.AdamW(params=self.agent.parameters(), lr=0.00002)

        if global_config.freeze_some_LM_layer:
            for name, param in self.agent.language_backbone.named_parameters():
                layer_num = re.findall("layer\.(\d+)\.", name)
                if len(layer_num) > 0 and int(layer_num[0]) > 4:
                    print("Unfreeze layer:", int(layer_num[0]))
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def infer_mask(self, generated_id_tensor):
        tmp = generated_id_tensor.detach().cpu().numpy().tolist()
        # mask_idx = [max(one.index(2), 10) if 2 in one else len(one) for one in tmp]
        mask_idx = [one.index(2) if 2 in one else len(one) for one in tmp]
        tmp_mask = [[1 if k <= mask_idx[i] else 0 for k, v in enumerate(j)] for i, j in enumerate(tmp)]
        return torch.tensor(tmp_mask).cuda()

    def masking_polarity_head(self, generated_id_tensor):
        tmp = generated_id_tensor.detach().cpu().numpy().tolist()
        mask_idx = [one.index(2) if 2 in one else len(one) for one in tmp]
        tmp_mask = [[1 if k <= mask_idx[i] and (k not in [1, 2]) else 0 for k, v in enumerate(j)] for i, j in enumerate(tmp)]

        if global_config.diversity_ctrl:
            tmp_pop_list = []
            for j in self.diversity_dict.keys():
                if self.diversity_dict[j] <= 1:
                    tmp_pop_list.append(j)
                else:
                    self.diversity_dict[j] -= 1

            for j in tmp_pop_list:
                self.diversity_dict.pop(j)

            self.diversity_pool = self.diversity_pool + tmp[:]

            if self.iter_step % 10 == 0:
                self.diversity_pool = self.diversity_pool[- global_config.batch_size * 10:]
                tmp_list = []
                [tmp_list.extend(list(set(i))) for i in self.diversity_pool]
                tmp_count_dict = Counter(tmp_list)
                for j in tmp_count_dict.keys():
                    if tmp_count_dict[j] > global_config.batch_size:
                        self.diversity_dict[j] = 100

                print("############", self.iter_step, self.agent.tokenizer.decode(self.diversity_dict.keys()), "############")

        tmp_mask_no_special_token = [[1 if k in [0, 1, 2, mask_idx[i], ] else 0 for k, v in enumerate(j)] for i, j in enumerate(tmp)]
        tmp_mask_diversity_mask = [[0.5 if v in self.diversity_dict.keys() and k not in [0, 1, 2, mask_idx[i], ] else 1.0 for k, v in enumerate(j)] for i, j in enumerate(tmp)]

        return torch.tensor(tmp_mask).cuda().contiguous(), torch.tensor(tmp_mask_no_special_token).cuda().contiguous(), torch.tensor(tmp_mask_diversity_mask).cuda().contiguous()

    def forward(self, batch, eval_mode=False):
        """ read and process data """
        batch_sample_input_text = [i[0] for i in batch]
        batch_input_style_list = [i.split()[0].replace("<s>", "") for i in batch_sample_input_text]

        batch_supervised_target_text = [i[1] for i in batch]
        batch_target_style_list = [i.split()[0].replace("<s>", "") for i in batch_supervised_target_text]

        """ build input tensors """
        batch_encoder_input = self.agent.tokenizer(batch_sample_input_text, return_tensors='pt', padding=True, add_special_tokens=False)
        batch_input_tensor = batch_encoder_input.data["input_ids"].cuda()
        batch_attention_mask = batch_encoder_input.data["attention_mask"].cuda()

        """ build decoder tensors for training, add </s> for the decoder ids for BART """
        batch_decoder_input = self.agent.tokenizer(batch_supervised_target_text, return_tensors='pt', padding=True, add_special_tokens=False)
        batch_decoder_label_tensor = batch_decoder_input.data["input_ids"].cuda()
        batch_decoder_attention_mask = batch_decoder_input.data["attention_mask"].cuda()

        batch_style_mask_0 = torch.tensor([1 - self.style_class_dict[i] for i in batch_target_style_list]).cuda().unsqueeze(1)
        batch_style_mask_1 = torch.tensor([self.style_class_dict[i] for i in batch_target_style_list]).cuda().unsqueeze(1)

        supervised_loss = None
        cyclic_loss = None
        GAN_discriminator_loss, GAN_generator_loss = None, None
        binary_cls_res = None
        style_average_lambda = None

        """ step 1 supervised training """
        if (self.RL_training is False) or (global_config.pure_unsupervised_training is False):
            """ teacher forcing """
            if self.teacher_forcing_rate == 1.0 or random.randint(0, 100) < 100 * self.teacher_forcing_rate:
                supervised_logits, supervised_loss = self.agent.rewrite_sentence_supervised(batch_input_tensor, batch_attention_mask, batch_decoder_label_tensor)
            else:
                _, _, output_greedy_logp, _, _, _, _, _, = self.agent.rewrite_sentence_RL(batch_input_tensor, batch_attention_mask, plus_sampling_generation=False)
                min_length = min(output_greedy_logp.size(1), batch_decoder_label_tensor.size(1))
                tmp_nll_loss = - output_greedy_logp[:, :min_length, :].gather(dim=-1, index=batch_decoder_label_tensor[:, :min_length].unsqueeze(2))
                tmp_nll_loss = tmp_nll_loss.squeeze(2) * batch_decoder_attention_mask[:, :min_length]
                supervised_loss = torch.mean(torch.sum(tmp_nll_loss, dim=1) / torch.sum(batch_decoder_attention_mask[:, :min_length], dim=1))

        """ step 2 self-critical training with rewards """
        if self.RL_training and not eval_mode:
            trans_sen_ids_greedy, trans_sen_text_greedy, trans_logp_greedy, trans_onehot_greedy, trans_sen_ids_sampling, trans_sen_text_sampling, trans_logp_sampling, trans_onehot_sampling \
                = self.agent.rewrite_sentence_RL(batch_input_tensor, batch_attention_mask, greedy_generation=False, plus_sampling_generation=True)

            logp_sample = torch.sum(trans_logp_sampling * trans_onehot_sampling, dim=-1)
            infer_mask = self.infer_mask(trans_sen_ids_sampling)

            """ Back Translation Training """
            """ here the reward should have the dim (batch_size, ) """
            with torch.no_grad():
                # cyclic_advantage = self.agent.cyclic_generation(batch_target_style_list, trans_sen_text_sampling, batch_sample_input_text) - \
                #                    self.agent.cyclic_generation(batch_target_style_list, trans_sen_text_greedy, batch_sample_input_text)

                cyclic_advantage = self.agent.cyclic_generation(batch_target_style_list, trans_sen_text_sampling, batch_sample_input_text)

            """ Style Discriminator Training """
            if trans_sen_text_greedy is not None:
                GAN_gen_sample_list_greedy = [i.replace("<pad>", "") for i in trans_sen_text_greedy]
            GAN_gen_sample_list_sampling = [i.replace("<pad>", "") for i in trans_sen_text_sampling]

            if self.iter_step % global_config.batch_loss_print_interval == 0:
                print(batch_sample_input_text[:2])
                if trans_sen_text_greedy is not None:
                    print(GAN_gen_sample_list_greedy[:2])
                print(GAN_gen_sample_list_sampling[:2])
                # print(batch_supervised_target_text[:2], "\n")

            """ using binary style classification  """
            with torch.no_grad():
                style_CLS_logits, _ = self.style_CLS.binary_cls(GAN_gen_sample_list_sampling)
                GAN_generator_advantage = torch.gather(torch.softmax(style_CLS_logits.detach(), dim=1), dim=1, index=batch_style_mask_1).detach()
                GAN_generator_advantage = torch.clamp(GAN_generator_advantage, max=0.8)

                style_average_lambda = torch.mean(GAN_generator_advantage).detach()

                """ add token level reward constrain """
                LM_target = trans_sen_ids_sampling.cuda().contiguous()
                LM_input_mask, post_mask, diversity_mask = self.masking_polarity_head(trans_sen_ids_sampling)

                attention_score_list = self.style_CLS.style_classifier(input_ids=LM_target, attention_mask=LM_input_mask).attentions

                tmp_matrix = [torch.max(attention_score_list[i], dim=1).values[:, 0, :].detach().cpu().numpy() for i in [-1, -2]]
                tmp_matrix = torch.tensor(tmp_matrix).cuda()
                token_level_salience = torch.max(tmp_matrix, dim=0).values

                lower_clip, higher_clip = 0.2, 0.5
                token_level_salience_wo_diversity = torch.clamp(token_level_salience, min=lower_clip, max=higher_clip)

                token_level_salience_with_diversity = torch.clamp(torch.clamp(token_level_salience, max=higher_clip) * diversity_mask, min=lower_clip)
                token_level_salience_with_diversity = (token_level_salience_with_diversity * (1 - post_mask)) + lower_clip * post_mask

                GAN_generator_advantage = GAN_generator_advantage * token_level_salience_with_diversity

            """ Here we re-weight the style advantage """
            if global_config.cyclic_balance:
                J_GAN = logp_sample * GAN_generator_advantage * cyclic_advantage.unsqueeze(1)
            else:
                J_GAN = logp_sample * GAN_generator_advantage

            assert J_GAN.dim() == 2
            GAN_generator_loss = - torch.sum(J_GAN * infer_mask) / torch.sum(infer_mask)

            """ here we also re-weight the cyclic advantage """
            # print(cyclic_advantage)
            with torch.no_grad():
                token_level_salience_for_cyclic = (lower_clip + higher_clip) - token_level_salience_wo_diversity
                cyclic_advantage_balanced = cyclic_advantage.unsqueeze(1) * token_level_salience_for_cyclic

            J_Cyclic = logp_sample * cyclic_advantage_balanced
            assert J_Cyclic.dim() == 2
            cyclic_loss = - torch.sum(J_Cyclic * infer_mask) / torch.sum(infer_mask)

        transferred_sen_text = None
        eval_beam_num = 1
        if eval_mode:
            if eval_beam_num == 1:
                output_sen_token_ids_eval, _ = self.agent.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask,
                                                                                     num_beams=1, min_length=10, max_length=30, early_stopping=False, use_cache=False)
            else:
                output_sen_token_ids_eval = self.agent.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask,
                                                                                  num_beams=eval_beam_num, min_length=7, max_length=30, early_stopping=False, use_cache=False)

            transferred_sen_text = [self.agent.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in output_sen_token_ids_eval]

            style_CLS_logits, style_CLS_pred = self.style_CLS.binary_cls(transferred_sen_text)
            binary_cls_res = [style_CLS_pred.detach().cpu().numpy().tolist(), batch_style_mask_1.detach().cpu().numpy().tolist()]

        return supervised_loss, cyclic_loss, GAN_discriminator_loss, GAN_generator_loss, transferred_sen_text, binary_cls_res, style_average_lambda

    def infer_forward(self, batch):
        batch_sample_input_text = [i[0] for i in batch]
        """ build input tensors """
        encoded_input = self.agent.tokenizer(batch_sample_input_text, return_tensors='pt', padding=True, add_special_tokens=False)
        batch_input_tensor = encoded_input.data["input_ids"].cuda()
        batch_attention_mask = encoded_input.data["attention_mask"].cuda()

        with torch.no_grad():
            output_sen_token_ids_eval = self.agent.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask,
                                                                              num_return_sequences=8, num_beam_groups=8, diversity_penalty=2.0, num_beams=8,
                                                                              min_length=10, max_length=40, early_stopping=False, no_repeat_ngram_size=3)

            transferred_sen_text = [self.agent.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in output_sen_token_ids_eval]

            _, style_CLS_pred = self.style_CLS.binary_cls(transferred_sen_text)
            cls_label = [1 - self.style_class_dict[i.split()[0].replace("<s>", "")] for i in batch_sample_input_text]
            binary_cls_res = [style_CLS_pred.detach().cpu().numpy().tolist(), cls_label]

        return transferred_sen_text, binary_cls_res

    def batch_train(self, batch, epoch_number):
        self.agent.train()
        self.optimizer_RL.zero_grad()
        self.optimizer_Supervised.zero_grad()

        if self.RL_training is False:
            supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, binary_cls_res, _ = self.forward(batch)
            (supervised_loss).backward()
            self.optimizer_Supervised.step()

        else:
            self.RL_training = True
            supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, binary_cls_res, style_average_lambda = self.forward(batch)
            supervised_loss = supervised_loss * self.supervised_loss_decay
            (supervised_loss + cyclic_loss + GAN_gen_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer_RL.step()

        self.iter_step += 1

        if supervised_loss:
            supervised_loss = supervised_loss.item()
        if cyclic_loss:
            cyclic_loss = cyclic_loss.item()
        if GAN_dis_loss:
            GAN_dis_loss = GAN_dis_loss.item()
        if GAN_gen_loss:
            GAN_gen_loss = GAN_gen_loss.item()

        return supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, binary_cls_res

    def batch_eval(self, batch):
        self.agent.eval()
        supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, binary_cls_res, _ = self.forward(batch, eval_mode=True)
        if supervised_loss:
            supervised_loss = supervised_loss.item()
        if cyclic_loss:
            cyclic_loss = cyclic_loss.item()
        if GAN_dis_loss:
            GAN_dis_loss = GAN_dis_loss.item()
        if GAN_gen_loss:
            GAN_gen_loss = GAN_gen_loss.item()

        return supervised_loss, cyclic_loss, GAN_dis_loss, GAN_gen_loss, transferred_sen_text, binary_cls_res

    def batch_infer(self, batch):
        self.agent.eval()
        transferred_sen_text, binary_cls_res = self.infer_forward(batch)
        return transferred_sen_text, binary_cls_res

    def save_model(self, save_path):
        """ save model """
        print("Saving model to:", save_path)
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, load_path):
        """ save model """
        print("Loading model from:", load_path)
        self.agent.load_state_dict(torch.load(load_path))
