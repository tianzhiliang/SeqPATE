"""
    Teaching.
"""
import random
import os, sys
import copy

import numpy as np
import torch
import torch.nn as nn
import texar.torch as tx
from torch.autograd import Variable
import utils.tensor_utils as tensor_utils
import utils.time_utils as time_utils
import utils.data_utils as data_utils
import utils.math_utils as math_utils

class Teaching:
    def __init__(self, config_train, seed=None, batch_size=32, nsamples=0, \
        max_decoding_length=100, teacher_model_name_prefix="", \
        top_p=40, top_k=40, temperature=0.7, interactive=True, \
        device=None, teacher_num=1, merge_teachers_opt=1, output_dir="", \
        teacher_generate_mode=0):

        # Options
        if seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if config_train.teachers_on_cpu != 1 and torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)

        self.nsamples = nsamples # only works in generate_from_scratch
        self.batch_size = batch_size
        self.max_decoding_length = max_decoding_length
        self.teacher_model_name_prefix = teacher_model_name_prefix
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.interactive = interactive
        self.device = device
        self.teacher_num = teacher_num
        self.merge_teachers_opt = merge_teachers_opt
        self.debug = config_train.debug
        self.config_train = config_train
        self.vocab_size = -1
        self.tmp_cnt_for_debug = 0
        self.gpu_results_to_cpu_for_memory = True
        self.output_dir = output_dir
        self.teacher_generate_mode = teacher_generate_mode
        self.sparse_merge_to_save_memory_opt = 2 # 0: not save memory 1: save when teacher num=200; 2:save when teacher num=2k

        # Tools
        # Create a GPT-2 tokenizer (BPE encoding)
        self.tokenizer = tx.data.GPT2Tokenizer(
            pretrained_model_name=self.teacher_model_name_prefix)
        self.end_token = self.tokenizer.map_token_to_id('<|endoftext|>')
        self.laplacian_factor = math_utils.Laplacian(mean=0, scale=1)
        #self.where_add_noise_opt = False # hard coding
        if self.config_train.pate_origin_noise_epsilon > 0:
            laplacian_noise_scale = 2.0 / (self.config_train.pate_origin_noise_epsilon \
                * self.config_train.teacher_num)
            if self.config_train.where_add_noise_opt == 1:
                vocab_size_for_laplacian_noise = 50257 # hard coding
            elif self.config_train.where_add_noise_opt in [2, 3]:
                if self.config_train.where_add_noise_opt == 2:
                    assert self.config_train.topk_for_memorize_prob == self.config_train.topk_for_noise
                vocab_size_for_laplacian_noise = self.config_train.topk_for_noise
            self.laplacian_noise = math_utils.Laplacian(mean=0, scale=laplacian_noise_scale, \
                device=self.device, use_cache=1, vocab_size=vocab_size_for_laplacian_noise, \
                batch_size=self.config_train.train_batch_size, max_seq_len=self.config_train.max_decoding_length)

        # Data
        self.teachers = []
        if self.config_train.noise_teacher_num > 0:
            self.load_dict_for_noise_sampling_when_init()

        # variables
        self.teacher_result_buf_num = config_train.teacher_label_batch // config_train.train_batch_size
        if 0 != config_train.teacher_label_batch % config_train.train_batch_size:
            print("error on teacher_label_batch (", config_train.teacher_label_batch, \
                ") and train_batch_size (", config_train.train_batch_size, ")")
        if self.debug >= 4:
            print("self.teacher_result_buf_num:", self.teacher_result_buf_num, \
                "config_train.teacher_label_batch:", config_train.teacher_label_batch, \
                "config_train.train_batch_size:", config_train.train_batch_size)
        self.teacher_result_buf_cnt = 0
        self.teacher_input_buf = [] # before teacher labeling
        self.teacher_result_buf = [] # after teacher labeling
        self.teacher_total_batchid_for_save_to_file = [0] * (self.config_train.teacher_num + 1) # is needed only when save_teacher_results_to_file==1
        self.queried_sample_token_dict = {}

        # functions
        self.nll_loss = nn.NLLLoss()
        #self.nn_kld_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False) # all are default values
        self.nn_kld_loss = nn.KLDivLoss()

    def init_all_teachermodels(self, teacher_checkpoint):
        for i in range(self.teacher_num):
            teacher_model = self.init_model(teacher_checkpoint + str(i))
            self.teachers.append(teacher_model)

    def init_all_teachermodels_to_cpu(self, teacher_checkpoint):
        # Build the teacher model
        for i in range(self.teacher_num):
            teacher_model = self.init_model_to_cpu(teacher_checkpoint + str(i))
            self.teachers.append(teacher_model)

    def load_dict_for_noise_sampling_when_init(self):
        training_file = self.config_train.train_hparam['dataset']['files']
        dictfile = os.path.dirname(training_file) + "/train.txt.dict"
        f = open(dictfile, "r")
        self.num2token_dict_for_noise_sampling = []
        cnt, last_cnt = 0, 0
        for line in f:
            line = line.strip().split("\t")
            token, freq = line
            token = torch.tensor(int(token), device=self.device)
            cnt += int(freq)
            for i in range(last_cnt, cnt):
                self.num2token_dict_for_noise_sampling.append(token)
            last_cnt = cnt
        f.close()

    def sample_batch_of_sents_as_noise(self, ori_batch):
        end_token = self.tokenizer.map_token_to_id('<|endoftext|>')
        if self.debug >= 10:
            print("in sample_batch_of_sents_as_noise end_token:", end_token)
        noise_batch = ori_batch.clone()
        if self.debug >= 10:
            print("before noise_batch:", noise_batch)
        for i in range(len(noise_batch)):
            for j in range(len(noise_batch[i])):
                if end_token == noise_batch[i][j].cpu().item():
                    continue
                noise_batch[i][j] = data_utils.sample_noise_from_vocab_with_cdf( \
                    self.num2token_dict_for_noise_sampling)
        if self.debug >= 10:
            print("after noise_batch:", noise_batch)
        return noise_batch

    def init_model_to_cpu(self, checkpoint):
        # Build the teacher model
        time_stat = time_utils.Time()
        time_stat.begin("load_gpt2_or_copy_gpt2")
        model = tx.modules.GPT2Decoder(self.teacher_model_name_prefix)
        #model = self.gpt_pretrain_model_for_init
        #model.load_state_dict({name: self.params_for_gpt_pretrain_model_for_init[name] \
        #    for name in self.params_for_gpt_pretrain_model_for_init})
        time_stat.end("load_gpt2_or_copy_gpt2")
        if checkpoint:
            time_stat.begin("load_checkpoint")
            ckpt = torch.load(checkpoint)
            time_stat.end("load_checkpoint")
            if self.debug >= 12:
                print("ckpt:", ckpt)
            #model.load_state_dict(ckpt['model'])
            time_stat.begin("load_state_dict")
            model.load_state_dict(ckpt)
            time_stat.end("load_state_dict")

        if self.debug >= 5:
            time_stat.print_all()

        if self.max_decoding_length > model.hparams.position_size:
            raise ValueError(
                "max_decoding_length should not be greater than position size")
        print("\nFinished loading\n")

        return model

    def init_model_copy_from_cpu_to_gpu_and_return(self, teacher_id):
        if self.config_train.teachers_on_cpu == 1:
            return

        #time_stat.begin("load_model_tocuda")
        #self.teachers[teacher_id] = self.teachers[teacher_id].detach()
        self.teachers[teacher_id].to(self.device)
        #self.teachers[teacher_id] = self.teachers[teacher_id].detach()
        torch.cuda.empty_cache()
        #time_stat.end("load_model_tocuda")

        #return self.teachers[teacher_id].detach()
        return self.teachers[teacher_id]

    def gpu_results_to_cpu_and_free_gpu_cache(self, v1):
        if v1 is not None:
            v1 = v1.cpu()
        #torch.cuda.empty_cache()
        return v1

    def gpu_results_to_cpu_and_free_gpu_cache_all(self, v1, v2, v3, v4):
        v1 = self.gpu_results_to_cpu_and_free_gpu_cache(v1)
        v2 = self.gpu_results_to_cpu_and_free_gpu_cache(v2)
        v3 = self.gpu_results_to_cpu_and_free_gpu_cache(v3)
        v4 = self.gpu_results_to_cpu_and_free_gpu_cache(v4)
        torch.cuda.empty_cache()
        return v1, v2, v3, v4

    def free_model_from_gpu(self, teacher_id):
        if self.config_train.teachers_on_cpu == 1:
            return

        #time_stat.begin("load_model_tocuda")
        self.teachers[teacher_id] = self.teachers[teacher_id].cpu()
        torch.cuda.empty_cache()
        #time_stat.end("load_model_tocuda")

    def init_model(self, checkpoint):
        # Build the teacher model
        time_stat = time_utils.Time()
        time_stat.begin("load_gpt2_or_copy_gpt2")
        model = tx.modules.GPT2Decoder(self.teacher_model_name_prefix)
        #model = self.gpt_pretrain_model_for_init
        #model.load_state_dict({name: self.params_for_gpt_pretrain_model_for_init[name] \
        #    for name in self.params_for_gpt_pretrain_model_for_init})
        time_stat.end("load_gpt2_or_copy_gpt2")
        if checkpoint:
            time_stat.begin("load_checkpoint")
            ckpt = torch.load(checkpoint)
            time_stat.end("load_checkpoint")
            if self.debug >= 12:
                print("ckpt:", ckpt)
            #model.load_state_dict(ckpt['model'])
            time_stat.begin("load_state_dict")
            model.load_state_dict(ckpt)
            time_stat.end("load_state_dict")

        if self.config_train.teachers_on_cpu != 1:
            time_stat.begin("load_model_tocuda")
            model.to(self.device)
            time_stat.end("load_model_tocuda")

        if self.debug >= 5:
            time_stat.print_all()

        if self.max_decoding_length > model.hparams.position_size:
            raise ValueError(
                "max_decoding_length should not be greater than position size")
        print("\nFinished loading\n")

        return model

    def _get_helper(self, start_tokens):
        if self.top_p:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.end_token,
                p=self.top_p,
                softmax_temperature=self.temperature)
        else:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.end_token,
                top_k=self.top_k,
                softmax_temperature=self.temperature)
        return helper

    def save_teacher_results_to_files(self, ids, lengths, t_logits_v, t_logits_i, vocab_size, teacher_id=0, is_save_data_when_teacher0=True):
        """
           t_logits_i, t_logits_v: index & value of topk results from teacher
        """

        if self.config_train.save_teacher_results_to_file == 0:
            return None
        if self.teacher_generate_mode == 0:
            return None

        file_id = self.teacher_total_batchid_for_save_to_file[teacher_id]
        print("in save_teacher_results_to_files. teacher_id:", teacher_id, "file_id:", file_id)
        print("save_teacher_results_to_files teacher_total_batchid_for_save_to_file:", self.teacher_total_batchid_for_save_to_file)
        self.teacher_total_batchid_for_save_to_file[teacher_id] += 1

        output_file = self.output_dir + "/teacher_generate_dist_pt." \
            + str(teacher_id) + "." + str(file_id)
        data = {"values": t_logits_v, "indexs": t_logits_i}
        torch.save(data, output_file)

        if teacher_id == 0 and is_save_data_when_teacher0:
            teacher_id_for_data = self.config_train.teacher_num
            file_id = self.teacher_total_batchid_for_save_to_file[teacher_id_for_data]
            self.teacher_total_batchid_for_save_to_file[teacher_id_for_data] += 1

            output_file = self.output_dir + "/teacher_generate_dist_pt." \
                + str(teacher_id_for_data) + "." + str(file_id)
            data = {"ids": ids, "lengths": lengths, "vocab_size": vocab_size}
            torch.save(data, output_file)

    def load_teacher_results_from_files_for_student(self):
        if self.config_train.save_teacher_results_to_file == 0:
            return None
        if self.teacher_generate_mode == 1:
            return None

        teachers_sample_id = [None] * self.config_train.teacher_num
        teachers_logits = [None] * self.config_train.teacher_num
        t_logits_v_list, t_logits_i_list = [], []
        for teacher_id in range(self.config_train.teacher_num):
            file_id = self.teacher_total_batchid_for_save_to_file[teacher_id]
            self.teacher_total_batchid_for_save_to_file[teacher_id] += 1

            result_file = self.config_train.teacher_generated_results + "/teacher_generate_dist_pt." \
                + str(teacher_id) + "." + str(file_id)
            if not os.path.exists(result_file):  # means end of this epoch
                self.teacher_total_batchid_for_save_to_file = [0] \
                    * (self.config_train.teacher_num + 1) # is needed only when save_teacher_results_to_file==1
                print("cannot find result_file:", result_file, "this epoch terminate. continue to next epoch")
                return [[None]*6]
            data = torch.load(result_file)
            t_logits_v = data['values']
            t_logits_i = data['indexs'].long()
            t_logits_v_list.append(t_logits_v)
            t_logits_i_list.append(t_logits_i)

        teacher_id_for_data = self.config_train.teacher_num
        file_id = self.teacher_total_batchid_for_save_to_file[teacher_id_for_data]
        self.teacher_total_batchid_for_save_to_file[teacher_id_for_data] += 1

        data_file = self.config_train.teacher_generated_results + "/teacher_generate_dist_pt." \
                + str(teacher_id_for_data) + "." + str(file_id)
        if torch.cuda.is_available():
            data = torch.load(data_file)
        else:
            data = torch.load(data_file, map_location=torch.device('cpu'))
        ids = data['ids']
        lengths = data['lengths']
        self.vocab_size = data['vocab_size']

        data_from_a_teacher_buf = [ids, lengths, teachers_sample_id, teachers_logits, \
                t_logits_v_list, t_logits_i_list]

        """
           Here, we do not consider teacher_label_batch.
           we only use train_batch_size and treat teacher_label_batch equal to train_batch_size
        """
        res = [data_from_a_teacher_buf]

        return res

    def teacher_collect_or_label(self, teacher_checkpoint, \
        input_ids, input_lengths):
        """
        INPUT:
            teacher_checkpoint: prefix of teacher checkpoint (str)
            input_ids: (train_batch_size, seq_len)
            input_lengths: (train_batch_size)
        OUTPUT:
            res 
            list of [input_ids, input_lengths, teachers_sample_id, teachers_logits]
            list length: self.teacher_result_buf_num
                input_ids [teacher_label_batch/train_batch, (train_batch, seq_len)]
                input_lengths [teacher_label_batch/train_batch, (train_batch)]
                teachers_sample_id [teacher_label_batch/train_batch, [teacher_num, (seq_len)]]
                teachers_logits [teacher_label_batch/train_batch, [teacher_num, (seq_len, vocab_size)]]
                t_logits_stopk_v_s: list of t_logits_stopk_v 
                    [teacher_label_batch/train_batch, [teacher_num, (train_batch, seq_len, topk_vocab)]]
                t_logits_stopk_i_s: list of t_logits_stopk_i 
                    [teacher_label_batch/train_batch, [teacher_num, (train_batch, seq_len, topk_vocab)]]
        """

        if self.config_train.save_teacher_results_to_file == 1 and self.teacher_generate_mode == 0: # teacher generate to file
            ret = self.load_teacher_results_from_files_for_student()
            return ret

        if 0 == self.teacher_result_buf_cnt:
            self.teacher_input_buf = []
        self.teacher_input_buf.append([input_ids, input_lengths])
        self.teacher_result_buf_cnt += 1

        if self.teacher_result_buf_cnt < self.teacher_result_buf_num: # collect
            if self.debug >= 6:
                print("self.teacher_result_buf_cnt:", self.teacher_result_buf_cnt)
            return None
        else: 
            res = []
            if self.debug >= 6:
                print("label self.teacher_result_buf_cnt:", self.teacher_result_buf_cnt)
            teachers_sample_id_tmp, teachers_logits_tmp, t_logits_stopk_v_list_tmp, \
                t_logits_stopk_i_list_tmp = [], [], [], []
            # init, collect, and predict
            time_stat = time_utils.Time()
            time_stat.begin("init_or_inference_teacher")
            t_sample_id, t_logits, t_logits_stopk_v, \
                t_logits_stopk_i = None,None,None,None
            for i in range(self.teacher_num):
            #for i in range(self.config_train.teacher_start_id, self.config_train.teacher_start_id + self.teacher_num):
            #for i in range(len(self.teachers)):
                time_stat.begin("init_teacher_model")
                #teacher_model = self.init_model(teacher_checkpoint + str(i))
                teacher_model = self.init_model_copy_from_cpu_to_gpu_and_return(i)
                time_stat.end("init_teacher_model")
                time_stat.begin("inference_teacher_model")
                tsi, tl, lstv, lsti = [], [], [], []
                for batch_id, ids_lengths in enumerate(self.teacher_input_buf):
                    ids, lengths = ids_lengths
                    #t_sample_id, t_logits, t_logits_stopk_v, \
                    #    t_logits_stopk_i = None,None,None,None
                    if self.config_train.topk_for_memorize_prob <= 0: # whole vocab, not topk vocab
                        if i >= self.teacher_num - self.config_train.noise_teacher_num: # noise teacher
                            ids = self.sample_batch_of_sents_as_noise(ids)
                        t_sample_id, t_logits, vocab_size = \
                            self._score_w_by_w_on_groundtruth_one_teacher(teacher_model, ids)
                    else:
                        #if i == 0:
                        if i >= self.teacher_num - self.config_train.noise_teacher_num: # noise teacher
                            ids = self.sample_batch_of_sents_as_noise(ids)
                        t_logits_stopk_v, t_logits_stopk_i, vocab_size \
                              = self._score_w_by_w_on_groundtruth_one_teacher( \
                              teacher_model, ids, self.config_train.topk_for_memorize_prob, with_softmax=True)
                    self.vocab_size = vocab_size
                    torch.cuda.empty_cache()

                    if self.gpu_results_to_cpu_for_memory:
                        t_sample_id, t_logits, t_logits_stopk_v, t_logits_stopk_i = \
                            self.gpu_results_to_cpu_and_free_gpu_cache_all(t_sample_id, \
                            t_logits, t_logits_stopk_v, t_logits_stopk_i)

                    if self.config_train.save_teacher_results_to_file == 1 and self.teacher_generate_mode == 1: # teacher generate to file
                        self.save_teacher_results_to_files(ids, lengths, t_logits_stopk_v, \
                            t_logits_stopk_i, vocab_size, teacher_id=i)
                        t_sample_id, t_logits, t_logits_stopk_v, t_logits_stopk_i = None,None,None,None
                        
                    tsi.append(t_sample_id)
                    tl.append(t_logits)
                    lstv.append(t_logits_stopk_v)
                    lsti.append(t_logits_stopk_i)
                self.free_model_from_gpu(i)
                time_stat.end("inference_teacher_model")

                teachers_sample_id_tmp.append(tsi)
                teachers_logits_tmp.append(tl)
                t_logits_stopk_v_list_tmp.append(lstv)
                t_logits_stopk_i_list_tmp.append(lsti)

            time_stat.end("init_or_inference_teacher")
            time_stat.begin("collect_and_merge_teacher_results")

            def _reset(self):
                self.teacher_input_buf = []
                self.teacher_result_buf_cnt = 0
                torch.cuda.empty_cache()
                if self.debug >= 5:
                    time_stat.print_all()

            # collect and merge 
            if self.config_train.save_teacher_results_to_file == 1 and self.teacher_generate_mode == 1: # teacher generate to file
                _reset(self)
                return None

            for bid, ids_lengths in enumerate(self.teacher_input_buf):
                ids, lengths = ids_lengths
                teachers_sample_id, teachers_logits, t_logits_stopk_v_list, \
                    t_logits_stopk_i_list = [], [], [], []

                for tid in range(self.teacher_num):
                    teachers_sample_id.append(teachers_sample_id_tmp[tid][bid])
                    teachers_logits.append(teachers_logits_tmp[tid][bid])
                    t_logits_stopk_v_list.append(t_logits_stopk_v_list_tmp[tid][bid])
                    t_logits_stopk_i_list.append(t_logits_stopk_i_list_tmp[tid][bid])
                res.append([ids, lengths, teachers_sample_id, teachers_logits, \
                    t_logits_stopk_v_list, t_logits_stopk_i_list])

                # TODO: free model and buffer from GPU
            time_stat.end("collect_and_merge_teacher_results")
            _reset(self)
            #print("res:", res)
            return res

    def merge_teachers_results(self, sample_ids_all, logits_all):
        """
            sample_ids_all: list of torch.tensor (list of top1 generated result)
            logits_all: list of torch.tensor (list of logits)
        """
        if self.merge_teachers_opt == 1: # top most
            pass
        elif self.merge_teachers_opt == 2: # merge probs
            logits_all_torch = torch.stack(logits_all)
            print("logits_all_torch.shape:", logits_all_torch.shape)
            logits_merge = logits_all_torch.sum(dim=0)
            print("logits_merge.shape:", logits_merge.shape)
            sample_ids = logits_merge.argmax(dim=-1)
            print("sample_ids.shape:", sample_ids.shape)
            print("sample_ids:", sample_ids)
            return sample_ids, logits_merge

    def score_sequence(self, sequence, index_output_seq):
        """
        if index_output_seq == len(sequence)-1: score on a single token
        else: score on a sequence (sequence[index_output_seq:])
        """
        pass

    def get_ts_loss_wo_ff(self, student_output, t_sample_id, t_logits, labels_for_stat=None, time_stat=None):
        """
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        t_sample_id (teacher_num, train_batch_size, seq_len)
        t_logits (teacher_num, train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        """

        if self.teacher_num == 1:
            avg_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                t_logits[0], student_output.sample_id[:, :-1], \
                student_output.logits[:, :-1, :], \
                labels_for_stat=labels_for_stat, time_stat=time_stat)
        elif self.teacher_num > 1:
            losses = []
            for i in range(self.teacher_num):
                for this_teacher_sample_id, this_teacher_logits in zip(t_sample_id, t_logits):
                    this_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                        this_teacher_logits, student_output.sample_id[:, :-1], \
                        student_output.logits[:, :-1, :], \
                        labels_for_stat=labels_for_stat, time_stat=time_stat)
                # clean current teacher's buf on gpu!!!

                losses.append(this_loss)
            avg_loss = torch.stack(losses).mean()
        else:
            print("error. self.teacher_num:", self.teacher_num)

        return avg_loss

    def get_ts_loss_with_topk_vocab(self, input_ids, student_output, \
        train_batchid, t_logits_stopk_v_s, t_logits_stopk_i_s, labels_for_stat=None, time_stat=None, step=-1):
        """
        input_ids (train_batch_size, seq_len)
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        t_logits_stopk_v_s: list of t_logits_stopk_v 
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        t_logits_stopk_i_s: list of t_logits_stopk_i ()
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        """

        if self.config_train.topk_for_memorize_prob == -1:
            return None

        if self.config_train.teacher_merge_opt == 1:
            return self.get_ts_loss_with_topk_vocab_merge_loss(input_ids, \
                student_output, train_batchid, t_logits_stopk_v_s, \
                t_logits_stopk_i_s, labels_for_stat=labels_for_stat, time_stat=time_stat)
        elif self.config_train.teacher_merge_opt == 2:
            return self.get_ts_loss_with_topk_vocab_merge_prob(input_ids, \
                student_output, train_batchid, t_logits_stopk_v_s, \
                t_logits_stopk_i_s, labels_for_stat=labels_for_stat, time_stat=time_stat, step=step)
        else:
            return None

    def get_ts_loss_with_topk_vocab_merge_loss(self, input_ids, student_output, \
        train_batchid, t_logits_stopk_v_s, t_logits_stopk_i_s, labels_for_stat=None, time_stat=None):
        """
        input_ids (train_batch_size, seq_len)
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        t_logits_stopk_v_s: list of t_logits_stopk_v 
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        t_logits_stopk_i_s: list of t_logits_stopk_i ()
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        if self.config_train.topk_for_memorize_prob == -1:
            return None
        """

        if self.teacher_num >= 1:
            losses = []
            for tid in range(self.teacher_num):
                print("len(t_logits_stopk_v_s):", len(t_logits_stopk_v_s))
                print("t_logits_stopk_v_s[0].shape:", t_logits_stopk_v_s[0].shape)
                print("t_logits_stopk_i_s[0].shape:", t_logits_stopk_i_s[0].shape)
                print("t_logits_stopk_v_s[0]:", t_logits_stopk_v_s[0])
                time_stat.begin("get_loss")
                if self.gpu_results_to_cpu_for_memory:
                    t_logits_stopk_v_s[tid] = t_logits_stopk_v_s[tid].cuda()
                    t_logits_stopk_i_s[tid] = t_logits_stopk_i_s[tid].cuda()
                t_logits_dense = tensor_utils.sparse_topk_to_dense_3D( \
                        t_logits_stopk_v_s[tid], t_logits_stopk_i_s[tid], self.vocab_size, with_softmax=True,\
                        device=self.device, time_stat=time_stat) 
                this_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits_dense, student_output.sample_id[:, :-1], student_output.logits[:, :-1, :])
                time_stat.end("get_loss")

                # clean current teacher's buf on gpu!!!
                time_stat.begin("empty_cache")
                torch.cuda.empty_cache()
                time_stat.end("empty_cache")
                if self.debug >= 6:
                    print("this_loss:", this_loss)
                losses.append(this_loss)

            #avg_loss = torch.stack(losses).mean() 
            avg_loss = self.tool_merge_teacher_prob_or_loss(losses, merge_dim=0) # TODO: to be tested 
            if self.debug >= 6:
                print("losses:", losses)
                print("avg_loss:", avg_loss)
        else:
            print("error. self.teacher_num:", self.teacher_num, "len(self.teachers):", len(self.teachers))

        return avg_loss

    def tool_online_merge_teacher_prob_or_loss(self, previous_prob_or_loss, cur_prob_or_loss, cur_cnt, time_stat=None):
        if not (self.config_train.use_laplacian_factor_on_noise == 0 \
            or self.config_train.noise_teacher_num == 0):
            return None

        # current setting (noise_teacher_num==0, use_laplacian_factor_on_noise==0)
            #return torch.stack(prob_or_loss_set).mean(dim=merge_dim)
        if previous_prob_or_loss is None and cur_cnt == 0:
            previous_prob_or_loss = cur_prob_or_loss
        else:
            previous_prob_or_loss = previous_prob_or_loss * cur_cnt
            previous_prob_or_loss += cur_prob_or_loss
            previous_prob_or_loss /= (cur_cnt + 1)
            torch.cuda.empty_cache()
            #previous_prob_or_loss = (previous_prob_or_loss * cur_cnt + cur_prob_or_loss) / (cur_cnt + 1)
        cur_cnt += 1
        return previous_prob_or_loss, cur_cnt

    def tool_merge_teacher_prob_or_loss(self, prob_or_loss_set, merge_dim=0, time_stat=None):
        def __get_mean_list_of_tensor_costing_less_memory(list_of_tensor, merge_dim):
            batch = 20
            tmp_res = []

            iter_num = len(list_of_tensor) // batch
            #print("len(list_of_tensor):", len(list_of_tensor), "iter_num:", iter_num)
            if len(list_of_tensor) % batch != 0:
                iter_num += 1
            iter_num = int(iter_num)
            for i in range(iter_num):
                begin = i * batch
                end = (i + 1) * batch
                if begin > len(list_of_tensor) - 1:
                    begin = len(list_of_tensor) - 1
                if end > len(list_of_tensor):
                    end = len(list_of_tensor)
                value = torch.stack(list_of_tensor[begin:end]).mean(dim=merge_dim)
                tmp_res.append(value)

            res = torch.stack(tmp_res).mean(dim=merge_dim)
            return res

        #self.where_add_noise_opt = False
        if self.config_train.pate_origin_noise_epsilon > 0 and self.config_train.where_add_noise_opt == 1:
            avg = torch.stack(prob_or_loss_set).mean(dim=merge_dim)
            #avg_w_noise = self.laplacian_noise.sample_and_add_noise_on(avg) 
            #indexs_for_noise = tensor_utils.union_set_of_list_of_ids(t_logits_stopk_i_s, device=self.device)
            avg_w_noise = self.laplacian_noise.sample_and_add_noise_on_with_cache(avg, time_stat=time_stat) 
            if self.debug >= 11:
                print("noise_from_laplacian before >=0. avg_w_noise:", avg_w_noise)
            #avg_w_noise = tensor_utils.set_valueA_where_src_smaller_valueB(\
            #    avg_w_noise, avg_w_noise, 0.0, 0.0)
            if self.debug >= 11:
                print("noise_from_laplacian. avg_real_teacher:", avg, \
                    "avg_w_noise:", avg_w_noise)
            return avg_w_noise
        elif self.config_train.use_laplacian_factor_on_noise == 0 \
            or self.config_train.noise_teacher_num == 0:
            #return torch.stack(prob_or_loss_set).mean(dim=merge_dim)
            return __get_mean_list_of_tensor_costing_less_memory(prob_or_loss_set, merge_dim) 
        elif self.config_train.use_laplacian_factor_on_noise == 1:
            laplacian_factor = self.laplacian_factor.sample_with_abs()
            real_teacher_num = self.config_train.teacher_num - self.config_train.noise_teacher_num
            avg_of_real_teacher = torch.stack(prob_or_loss_set[:real_teacher_num]).mean(dim=merge_dim)
            avg_of_noise_teacher = torch.stack(prob_or_loss_set[real_teacher_num:]).mean(dim=merge_dim)
            avg_of_noise_teacher2 = avg_of_noise_teacher * laplacian_factor
            avg = (avg_of_real_teacher * real_teacher_num \
                 + avg_of_noise_teacher2 * self.config_train.noise_teacher_num) \
                 / self.config_train.teacher_num
            if self.debug >= 11:
                print("merge_noise_teacher_w_laplacian. laplacian_factor:", \
                    laplacian_factor, "avg_real_teacher:", avg_of_real_teacher, \
                    "avg_noise_teacher:", avg_of_noise_teacher, \
                    "avg_noise_teacher after laplacian:", avg_of_noise_teacher2, \
                    "final_avg:", avg)
            if self.debug >= 17:
                print("merge_noise_teacher_w_laplacian. laplacian_factor:", laplacian_factor)
            return avg
        else:
            return None

    def get_ts_loss_with_topk_vocab_merge_prob(self, input_ids, student_output, \
        train_batchid, t_logits_stopk_v_s, t_logits_stopk_i_s, labels_for_stat=None, time_stat=None, step=-1):
        """
        input_ids (train_batch_size, seq_len)
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        t_logits_stopk_v_s: list of t_logits_stopk_v 
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        t_logits_stopk_i_s: list of t_logits_stopk_i ()
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        """

        if self.teacher_num >= 1:
            t_logits_denses = []
            t_logits_dense_cumulate = None
            t_logits_dense_cumulate_cnt = 0
            for tid in range(self.teacher_num):
                if self.debug >= 7:
                    print("len(t_logits_stopk_v_s):", len(t_logits_stopk_v_s))
                    print("t_logits_stopk_v_s[0].shape:", t_logits_stopk_v_s[0].shape)
                    print("t_logits_stopk_i_s[0].shape:", t_logits_stopk_i_s[0].shape)
                if self.debug >= 10:
                    print("t_logits_stopk_v_s[0]:", t_logits_stopk_v_s[0])
                time_stat.begin("get_loss")
                if self.gpu_results_to_cpu_for_memory:
                    t_logits_stopk_v_s[tid] = t_logits_stopk_v_s[tid].cuda()
                    t_logits_stopk_i_s[tid] = t_logits_stopk_i_s[tid].cuda()
                #self.where_add_noise_opt = False
                if self.config_train.pate_origin_noise_epsilon > 0 and self.config_train.where_add_noise_opt == 2:
                    d_w_noise = self.laplacian_noise.sample_and_add_noise_on_with_cache(t_logits_stopk_v_s[tid], time_stat=time_stat)
                    t_logits_stopk_v_s[tid] = d_w_noise

                t_logits_dense = tensor_utils.sparse_topk_to_dense_3D( \
                        t_logits_stopk_v_s[tid], t_logits_stopk_i_s[tid], self.vocab_size, with_softmax=True,\
                        device=self.device, time_stat=time_stat) 
                if self.sparse_merge_to_save_memory_opt == 2:
                    t_logits_dense_cumulate, t_logits_dense_cumulate_cnt = \
                        self.tool_online_merge_teacher_prob_or_loss( \
                            t_logits_dense_cumulate, t_logits_dense, t_logits_dense_cumulate_cnt)
                else:
                    t_logits_denses.append(t_logits_dense)
                time_stat.end("get_loss")

                # clean current teacher's buf on gpu!!!
                time_stat.begin("empty_cache")
                torch.cuda.empty_cache()
                time_stat.end("empty_cache")

            if self.sparse_merge_to_save_memory_opt == 0:
                t_logits_dense_merged = torch.stack(t_logits_denses).mean(dim=0) 
            elif self.sparse_merge_to_save_memory_opt == 2:
                t_logits_dense_merged = t_logits_dense_cumulate
            elif self.sparse_merge_to_save_memory_opt == 1:
                print("len t_logits_denses:", len(t_logits_denses))
                print("t_logits_denses[0].shape:", t_logits_denses[0].shape)
                t_logits_dense_merged = self.tool_merge_teacher_prob_or_loss( \
                        t_logits_denses, merge_dim=0, time_stat=time_stat)

            avg_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits_dense_merged, student_output.sample_id[:, :-1], \
                    student_output.logits[:, :-1, :], labels_for_stat=labels_for_stat, time_stat=time_stat, train_batchid=train_batchid, step=step)
            if self.debug >= 6:
                print("merge prob mode avg_loss:", avg_loss)
        else:
            print("error. self.teacher_num:", self.teacher_num, "len(self.teachers):", len(self.teachers))

        return avg_loss

    def get_ts_loss_with_topk_vocab0(self, input_ids, student_output):
        """
        input_ids (train_batch_size, seq_len)
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        """

        if self.config_train.topk_for_memorize_prob == -1:
            return None

        if self.teacher_num == 1:
            t_logits_stopk_v, t_logits_stopk_i, vocab_size \
                    = self._score_w_by_w_on_groundtruth_one_teacher( \
                    self.teachers[0], input_ids, self.config_train.topk_for_memorize_prob)
            avg_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits, student_output.sample_id[:, :-1], student_output.logits[:, :-1, :])
        elif len(self.teachers) > 1:
            losses = []
            for i in range(len(self.teachers)):
                t_logits_stopk_v, t_logits_stopk_i, vocab_size \
                    = self._score_w_by_w_on_groundtruth_one_teacher( \
                    self.teachers[i], input_ids, self.config_train.topk_for_memorize_prob)
                if self.config_train.teacher_merge_opt == 1:
                    t_logits_dense = tensor_utils.sparse_topk_to_dense( \
                        t_logits_stopk_v, t_logits_stopk_i, vocab_size) # TODO:vocab_size
                #loss_data, avg_loss = self._ts_loss_w_by_w_on_gt( \
                this_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits_dense, student_output.sample_id[:, :-1], student_output.logits[:, :-1, :])
                # clean current teacher's buf on gpu!!!

                losses.append(this_loss)
            avg_loss = torch.stack(losses).mean()
        else:
            print("error. self.teacher_num:", self.teacher_num, "len(self.teachers):", len(self.teachers))

        return avg_loss

    def get_ts_loss(self, input_ids, student_output):
        """
        input_ids (train_batch_size, seq_len)
        student_output 
            student_output.sample_id (train_batch_size, seq_len)
            student_output.logits (train_batch_size, seq_len, vocabulary_size or topk_vocabulary_size)
        """

        if 1 == self.config_train.teachers_on_cpu:
            input_ids = input_ids.cpu()
            student_output = student_output.cpu()

        if len(self.teachers) == 1:
            t_sample_id, t_logits, vocab_size = self._score_w_by_w_on_groundtruth_one_teacher( \
                    self.teachers[0], input_ids)
            #loss_data, avg_loss = self._ts_loss_w_by_w_on_gt( \
            avg_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits, student_output.sample_id[:, :-1], student_output.logits[:, :-1, :])
        elif len(self.teachers) > 1:
            losses = []
            for i in range(len(self.teachers)):
                t_sample_id, t_logits, vocab_size = self._score_w_by_w_on_groundtruth_one_teacher( \
                    self.teachers[i], input_ids)
                #loss_data, avg_loss = self._ts_loss_w_by_w_on_gt( \
                this_loss = self._ts_loss_w_by_w_on_gt_by_kld_loss( \
                    t_logits, student_output.sample_id[:, :-1], student_output.logits[:, :-1, :])
                # clean current teacher's buf on gpu!!!

                losses.append(this_loss)
            avg_loss = torch.stack(losses).mean()

        if 1 == self.config_train.teachers_on_cpu:
            avg_loss.to(self.device)
        return avg_loss

    def _sharper_prob_with_temperatured_softmax(self, t_prob, dim_for_softmax=-1):
        """
           make the prob (teacher's prob) more sharper
           t_prob: merged teacher's prob (after softmax)  (batch, seqlen, vocabsize)

           output:
               new_t_prob: t_prob (more sharper) (batch, seqlen, vocabsize)
        """

        if self.debug >= 9:
            print("original t_prob in _sharper_prob_with_temperatured_softmax:", t_prob)
            print("original t_prob max in _sharper_prob_with_temperatured_softmax:", t_prob.max(dim=dim_for_softmax).values)

        t = torch.tensor(self.config_train.temperature_for_softmax_for_sharper_prob * 1.0, device=self.device)
        new_t_prob = torch.softmax(t_prob / t, dim=dim_for_softmax)

        if self.debug >= 9:
            print("t_prob max after _sharper_prob_with_temperatured_softmax:", new_t_prob.max(dim=dim_for_softmax).values)
            print("t_prob after _sharper_prob_with_temperatured_softmax:", new_t_prob)
        return new_t_prob

    def _add_noise_by_indexs(self, t_prob, s_topk_indexs, dense_ori_dim, time_stat=None):
        """
        add noise on merged teachers' prob over student's topk (work when where_add_noise_opt==3)
        t_prob: merged teacher's prob (after softmax)  (batch, seqlen, vocabsize)
        s_topk_indexs: index of student's topk (batch,seqlen,K)
        dense_ori_dim: vocabsize

        output:
            new_t_prob: t_prob with noise (batch, seqlen, vocabsize)
        """

        # Step1: get noise
        dim_of_real_index = -1
        target_shape = list(s_topk_indexs.shape[:dim_of_real_index]) + [dense_ori_dim]
        # dims_of_non_real_index = s_topk_indexs.shape[0] * s_topk_indexs.shape[1]
        dims_of_non_real_index = np.prod(list(s_topk_indexs.shape[:dim_of_real_index]))

        zero = torch.tensor(0.0, device=self.device, requires_grad=False)
        almost_zero = torch.tensor(1e-15, device=self.device, requires_grad=False)
        # zeros: (batch,seqlen,K); noise: (batch,seqlen,K)
        zeros = torch.zeros(s_topk_indexs.shape, device=self.device, requires_grad=False)
        noise = self.laplacian_noise.sample_and_add_noise_on_with_cache(zeros, time_stat=time_stat)
        if self.debug >= 13:
            print("zeros:", zeros)
            print("noise:", noise)
        
        # Step2: reshape (3D -> 2D)
        def __3D_to_2D_reshape(src, dim_of_real_index, mark=""):
            src_shape = src.shape
            print("__3D_to_2D_reshape", mark, "dim_of_real_index:", dim_of_real_index, \
                "src_shape:", src_shape, "dims_of_non_real_index:", dims_of_non_real_index)
            src_2D = src.reshape([dims_of_non_real_index, int(src_shape[dim_of_real_index])])
            print("src_2D.shape:", src_2D.shape)
            return src_2D

        # Step2.1: reshape s_topk_indexs (3D -> 2D)
        # s_topk_indexs: (batch, seqlen, K); s_topk_indexs2D:(batch*seqlen, K)
        s_topk_indexs2D = __3D_to_2D_reshape(s_topk_indexs, dim_of_real_index, "s_topk_index")

        # Step2.2: reshape t_prob (3D -> 2D)
        # t_prob: (batch, seqlen, vocabsize); t_prob2D:(batch*seqlen, vocabsize)
        t_prob2D = __3D_to_2D_reshape(t_prob, dim_of_real_index, "t_prob")

        # Step2.3: reshape noise (3D -> 2D)
        # noise: (batch, seqlen, K); noise2D:(batch*seqlen, K)
        noise2D = __3D_to_2D_reshape(noise, dim_of_real_index, "noise")

        # Step3: add noise on data
        if time_stat is not None:
            time_stat.begin("_add_noise_by_indexs add noise one-by-one")
        if dims_of_non_real_index != s_topk_indexs2D.shape[0]:
            print("error!!! dims_of_non_real_index != s_topk_indexs2D.shape[0]", \
                dims_of_non_real_index, s_topk_indexs2D.shape[0])
        K = s_topk_indexs2D.shape[dim_of_real_index]
        print("K in _add_noise_by_indexs:", K)
        if self.debug >= 13:
            print("s_topk_indexs2D:", s_topk_indexs2D)
            print("s_topk_indexs2D[0]:", s_topk_indexs2D[0])
            print("s_topk_indexs2D[00]:", s_topk_indexs2D[0][0])
        t_prob2D_old = t_prob2D.clone()
        skip_topk_due_to_threshold = 0
        for i in range(dims_of_non_real_index):
            for j in range(K):
                #cnt = K * i + j
                real_index = s_topk_indexs2D[i][j].int()
                if real_index == -1: # for _get_student_topk_and_mask_others_K_vary_with_sample (self.config_train.adaptive_topk_threshold>=0)
                    print("real_index == -1. i:", i, "j:", j)
                    skip_topk_due_to_threshold += 1
                    continue
                t_prob2D[i][real_index] += noise2D[i][j]
        diff_after_noise = (t_prob2D_old!= t_prob2D).int().sum()
        print("in _add_noise after diff_after_noise:", diff_after_noise, \
            "skip_topk_due_to_threshold:", skip_topk_due_to_threshold, "total:", dims_of_non_real_index*K)

        # t_prob2D_v2:(batch*seqlen, vocabsize)
        t_prob2D_v2 = tensor_utils.set_valueA_where_src_smaller_valueB(\
            t_prob2D, t_prob2D, almost_zero, almost_zero)
            #t_prob2D, t_prob2D, zero, zero)
        #t_prob2D_v2 = t_prob2D

        # new_t_prob: (batch, seqlen, vocabsize)
        new_t_prob = t_prob2D_v2.reshape(target_shape)
        if self.debug >= 13:
            print("t_prob2D:", t_prob2D)
            print("t_prob2D_v2:", t_prob2D_v2)
            print("new_t_prob:", new_t_prob)

        if time_stat is not None:
            time_stat.end("_add_noise_by_indexs add noise one-by-one")

        return new_t_prob

    def _get_student_topk_and_mask_others_K_vary_with_sample(self, s_prob, t_prob, K=100, labels_for_stat=None, train_batchid=-1, step=-1):
        """
        s_prob: student generated results prob (batch, seqlen, vocabsize)
        t_prob: teacher reference prob (batch, seqlen, vocabsize)
        K: topk (config_train.topk_for_noise)
        labels_for_stat: only for stat (no use for training)
        if the value is not student's topk, assign teacher's prob on the value (so as to kld loss=0)

        output:
          new_s_prob: student generated results after mask 
                            (batch, seqlen, vocabsize)
          s_topk_indexs: index of student's topk prob (batch,seqlen,K)
          dense_ori_dim: vocabsize
        """

        if self.config_train.adaptive_topk_threshold <= 0:
            return None

        # get topk_idx
        # topk_value: (batch,seqlen,K);  topk_idx:(batch,seqlen,K)
        topk_value, topk_idx, dense_ori_dim, _ = \
           tensor_utils.dense_to_topk_in_sparse(s_prob, K=K, with_softmax = True)

        if labels_for_stat is not None and self.debug >=6: 
            tensor_utils._count_gt_in_student_topk(labels_for_stat, topk_idx, device=self.device, mark="ori_topk")

        topk_value_coverage = topk_value.sum(dim=-1).mean()
        if self.debug >= 7:
            print("topk_value_coverage in _get_student_topk_and_mask_others:", topk_value_coverage.item())

        # calculate the adaptive K according to the threshold, and get new topk_idx (list of tensor)
        topk_real_K = tensor_utils._get_index_given_threshold_on_cdf_3D(topk_value, self.config_train.adaptive_topk_threshold, self.config_train.adaptive_topk_min_k, self.device)
        print("in _get_student topk_real_K:", topk_real_K)
        tensor_utils._count_topk_coverage_with_topk_real_K(topk_value, topk_real_K)

        # mask topk_idx 
        # topk_real_K (batch, seqlen), the value shows the real K
        # topk_idx (batch, seqlen, ori_K), the value shows the index (for ori K's version)
        # OUTPUT: topk_idx (batch, seqlen, ori_K), the value shows the index while -1 as the value shows do not use this slot
        topk_idx_old = topk_idx.clone()
        print("in _get_student before topk_idx:", topk_idx)
        tensor_utils.mask_topk_idx_with_topk_real_K(topk_idx, topk_real_K, self.device)
        print("in _get_student after topk_idx:", topk_idx)
        diff = (topk_idx_old != topk_idx).int().sum()
        print("in _get_student after diff:", diff, "total:", topk_idx_old.nelement())
        #self.__get_topk_idx_adaptively()
        if labels_for_stat is not None and self.debug >=6: 
            tensor_utils._count_gt_in_student_topk(labels_for_stat, topk_idx, device=self.device, mark="topk_with_real_K_threshold")

        # topk_idx -> masked tensor (masked tensor to show the topk_idx information)
        # mask: (batch, seqlen, vocabsize)  topk should be masked as 0, others 1
        one = torch.tensor(1.0, device=self.device)
        mask = tensor_utils.get_binary_mask_from_sparse_index_3D(topk_idx, \
            dense_ori_dim, value_of_index=one, time_stat=None, device=self.device)
        if self.debug >= 13:
            print("topk_value:", topk_value)
            print("mask:", mask)

        if self.config_train.query_drop_rate > 0:
            print("debug select_threshold_for_active_learning:", \
                self.config_train.select_threshold_for_active_learning)
            mask, topk_idx = tensor_utils._random_or_active_mask_1Dvector_in_3Dtensor( \
                mask, topk_idx, self.config_train.query_drop_rate, device=self.device, \
                labels_for_select=labels_for_stat, \
                select_threshold=self.config_train.select_threshold_for_active_learning,\
                train_batchid=train_batchid, queried_sample_token_dict=self.queried_sample_token_dict, \
                t_prob=t_prob, max_uniq_query_num=self.config_train.max_uniq_query_num)

        # apply mask upon data
        #new_s_prob = tensor_utils.set_value_where_src_eq_value_wo_clone(\
        #    mask, s_prob, value=one)
        # new_s_prob, s_prob, t_prob: (batch, seqlen, vocabsize)
        # topk idx: mask is 0   ->    s_prob is unchanged
        # non-topk idx: mask is 1   ->    s_prob is unchanged
        new_s_prob = tensor_utils.assign_tensorA_element_to_tensorB_element_where_masked( \
            t_prob, s_prob, mask, time_stat=None, \
            avoid_negative_value=True, device=self.device)

        if self.debug >= 13:
            print("new_s_prob:", new_s_prob)
            print("topk_idx:", topk_idx)

        return new_s_prob, topk_idx, dense_ori_dim

    def _get_student_topk_and_mask_others(self, s_prob, t_prob, K=100, labels_for_stat=None, train_batchid=-1, step=-1):
        """
        s_prob: student generated results prob (batch, seqlen, vocabsize)
        t_prob: teacher reference prob (batch, seqlen, vocabsize)
        K: topk (config_train.topk_for_noise)
        labels_for_stat: only for stat (no use for training) (batch, seqlen)
        if the value is not student's topk, assign teacher's prob on the value (so as to kld loss=0)

        output:
          new_s_prob: student generated results after mask 
                            (batch, seqlen, vocabsize)
          s_topk_indexs: index of student's topk prob (batch,seqlen,K)
          dense_ori_dim: vocabsize
        """

        if self.config_train.adaptive_topk_threshold > 0:
            return self._get_student_topk_and_mask_others_K_vary_with_sample(s_prob, \
                t_prob, K=K, labels_for_stat=labels_for_stat, train_batchid=train_batchid, step=step)

        # get topk_idx
        # topk_value: (batch,seqlen,K);  topk_idx:(batch,seqlen,K)
        topk_value, topk_idx, dense_ori_dim, _ = \
           tensor_utils.dense_to_topk_in_sparse(s_prob, K=K, with_softmax = True)

        if labels_for_stat is not None and self.debug >=6: 
            tensor_utils._count_gt_in_student_topk(labels_for_stat, topk_idx, device=self.device, mark="ori_topk")

        topk_value_coverage = topk_value.sum(dim=-1).mean()
        if self.debug >= 7:
            print("topk_value_coverage in _get_student_topk_and_mask_others:", topk_value_coverage)

        # topk_idx -> masked tensor (masked tensor to show the topk_idx information)
        # mask: (batch, seqlen, vocabsize)  topk should be masked as 0, others 1
        one = torch.tensor(1.0, device=self.device)
        mask = tensor_utils.get_binary_mask_from_sparse_index_3D(topk_idx, \
            dense_ori_dim, value_of_index=one, time_stat=None, device=self.device)
        if self.debug >= 13:
            print("topk_value:", topk_value)
            print("mask:", mask)

        if self.config_train.query_drop_rate > 0:
            print("debug select_threshold_for_active_learning:", \
                self.config_train.select_threshold_for_active_learning)
            mask, topk_idx = tensor_utils._random_or_active_mask_1Dvector_in_3Dtensor( \
                mask, topk_idx, self.config_train.query_drop_rate, device=self.device, \
                labels_for_select=labels_for_stat, \
                select_threshold=self.config_train.select_threshold_for_active_learning,\
                train_batchid=train_batchid, queried_sample_token_dict=self.queried_sample_token_dict, \
                step=step, t_prob=t_prob, max_uniq_query_num=self.config_train.max_uniq_query_num)

        # apply mask upon data
        #new_s_prob = tensor_utils.set_value_where_src_eq_value_wo_clone(\
        #    mask, s_prob, value=one)
        # new_s_prob, s_prob, t_prob: (batch, seqlen, vocabsize)
        new_s_prob = tensor_utils.assign_tensorA_element_to_tensorB_element_where_masked( \
            t_prob, s_prob, mask, time_stat=None, \
            avoid_negative_value=True, device=self.device)

        if self.debug >= 13:
            print("new_s_prob:", new_s_prob)
            print("topk_idx:", topk_idx)

        return new_s_prob, topk_idx, dense_ori_dim

    def _ts_loss_w_by_w_on_gt_by_kld_loss(self, t_logits, \
            s_sample_id, s_logits, t_logtis_is_t_prob_with_softmax=True, \
            labels_for_stat=None, time_stat=None, train_batchid=-1, step=-1):
        """
        t_logits: from one teacher or several teachers
        """

        t_logits = t_logits.detach()
        if t_logtis_is_t_prob_with_softmax:
            t_prob = t_logits
        else:
            t_prob = torch.softmax(t_logits, dim=-1)
        print("before s_logits.sum():", s_logits.sum(), "s_logits.mean():", s_logits.mean(), "s_logits.max:", s_logits.max(), "s_logits.min:", s_logits.min())
        s_prob = torch.softmax(s_logits, dim=-1)

        if self.config_train.where_add_noise_opt == 3:
            # s_prob, t_prob: (batch, seqlen, vocabsize)
            t_prob_old = t_prob.clone()
            s_prob_old = s_prob.clone()
            print("before s_prob.sum():", s_prob.sum(), "s_prob.mean():", s_prob.mean(), "s_prob.max:", s_prob.max(), "s_prob.min:", s_prob.min())
            if self.config_train.temperature_for_softmax_for_sharper_prob != 1.0 and \
                self.config_train.temperature_for_softmax_for_sharper_prob != 0.0:
                t_prob = self._sharper_prob_with_temperatured_softmax(t_prob, dim_for_softmax=-1)
            s_prob, s_topk_indexs, dense_ori_dim = self._get_student_topk_and_mask_others( \
                s_prob, t_prob, K=self.config_train.topk_for_noise, labels_for_stat=labels_for_stat, train_batchid=train_batchid, step=step)
            t_prob = self._add_noise_by_indexs(t_prob, s_topk_indexs, dense_ori_dim, time_stat=time_stat)
        s_prob_log = torch.log(s_prob)

        t_prob_f = t_prob.reshape(t_prob.shape[0] * t_prob.shape[1], t_prob.shape[-1])
        s_prob_log_f = s_prob_log.reshape(s_prob_log.shape[0] * s_prob_log.shape[1], s_prob_log.shape[-1])

        if self.debug >= 5 and tensor_utils.tensor_equal(s_prob_log, s_logits):
            print("s_prob_log == s_prob")
        if self.debug >= 5:
            print("t_logits.shape:", t_logits.shape)
            print("t_prob.shape:", t_prob.shape)
            print("s_logits.shape:", s_logits.shape)
            print("s_sample_id.shape:", s_sample_id.shape)
            print("t_prob_f.sum():", t_prob_f.sum(), "t_prob_f.mean():", t_prob_f.mean(), "t_prob_f.max:", t_prob_f.max(), "t_prob_f.min:", t_prob_f.min())
            print("s_prob.sum():", s_prob.sum(), "s_prob.mean():", s_prob.mean(), "s_prob.max:", s_prob.max(), "s_prob.min:", s_prob.min())
            print("s_prob_log_f.sum():", s_prob_log_f.sum(), "s_prob_log_f.mean():", s_prob_log_f.mean(), "s_prob_log_f.max:", s_prob_log_f.max(), "s_prob_log_f.min:", s_prob_log_f.min())
            #torch.set_printoptions(profile="full")
            #print("s_prob_log:", s_prob_log)
            #print("s_prob_log_f:", s_prob_log_f)
            #torch.set_printoptions(profile="default") # reset
        if self.debug >= 7:
            print("t_logits:", t_logits)
            print("t_prob:", t_prob)
            print("s_logits:", s_logits)
            print("s_prob:", s_prob)
            print("s_prob_log:", s_prob_log)
            print("t_prob_f:", t_prob_f)
            print("s_prob_log_f:", s_prob_log_f)

        kld_loss = self.nn_kld_loss(s_prob_log_f, t_prob_f)
        #if self.config_train.pate_nll_loss_instead_of_kl == 1:
        #    t_prob_f_max_index = t_prob_f.max(dim=1).indices
        #    if self.debug >= 7:
        #        print("t_prob_f_max_index:", t_prob_f_max_index)
        #        print("t_prob_f_max_index.shape:", t_prob_f_max_index.shape)
        #    kld_loss = self.nll_loss(s_prob_log_f, t_prob_f_max_index)

        # Only for print
        if self.config_train.where_add_noise_opt == 3:
            t_prob_f_old = t_prob_old.reshape(t_prob_old.shape[0] * t_prob_old.shape[1], t_prob_old.shape[-1])
            s_prob_log_old = torch.log(s_prob_old)
            s_prob_log_f_old = s_prob_log_old.reshape( \
                s_prob_log_old.shape[0] * s_prob_log_old.shape[1], s_prob_log_old.shape[-1])
            kld_loss_old = self.nn_kld_loss(s_prob_log_f_old, t_prob_f_old)

            if self.debug >= 5:
                print("normal kld_loss:", kld_loss_old.item(), "student_topk_kld_loss:", kld_loss.item())

        return kld_loss

    def _ts_loss_w_by_w_on_gt(self, t_sample_id, t_logits, s_sample_id, s_logits):
        """
        t_logits: from one teacher or several teachers
        """
        t_prob = torch.softmax(t_logits, dim=-1)
        if self.debug >= 5:
            print("t_logits.shape:", t_logits.shape)
            print("t_prob.shape:", t_prob.shape)
            print("t_sample_id.shape:", t_sample_id.shape)
            print("s_sample_id.shape:", s_sample_id.shape)
        if self.debug >= 7:
            print("t_logits:", t_logits)
            print("t_prob:", t_prob)
            print("t_sample_id:", t_sample_id)

        t_prob_f = t_prob.reshape(t_prob.shape[0] * t_prob.shape[1], t_prob.shape[-1])
        s_sample_id_f = s_sample_id.reshape(s_sample_id.shape[0] * s_sample_id.shape[1])
        if self.debug >= 5:
            print("t_prob_f.shape:", t_prob_f.shape)
            print("s_sample_id_f.shape:", s_sample_id_f.shape)

        #t_prob_on_s_sample_f = t_prob_f[s_sample_id_f]
        #print("t_prob_on_s_sample_f.shape:", t_prob_on_s_sample_f.shape)

        t_prob_bufs = [Variable(torch.tensor(0.0), requires_grad=False)] * s_sample_id_f.shape[0]
        for i, (ss, tp) in enumerate(zip(s_sample_id_f, t_prob_f)):
            if self.debug >= 8:
                print("ss:", ss, "tp.shape:", tp.shape)
                print("ss:", ss, "tp:", tp)
            t_prob_bufs[i] = tp[ss]
        t_prob_on_s_sample_f = torch.stack(t_prob_bufs)
        if self.debug >= 5:
            print("t_prob_on_s_sample_f.shape:", t_prob_on_s_sample_f.shape)

        t_prob_on_s_sample = t_prob_on_s_sample_f.reshape(t_prob.shape[0], t_prob.shape[1])
        if self.debug >= 5:
            print("t_prob_on_s_sample.shape:", t_prob_on_s_sample.shape)

        losses = []
        for t_sample, t_p, s_sample, s_log in zip(t_sample_id, t_prob_on_s_sample, s_sample_id, s_logits):
            loss_this_sample = []
            for ts, tp, ss, sl in zip(t_sample, t_p, s_sample, s_log):
                if self.debug >= 8:
                    print("ts.unsqueeze(0):", ts.unsqueeze(0).unsqueeze(0), \
                        "tp:", tp, "ss:", ss, "sl.unsqueeze(0):", sl.unsqueeze(0).unsqueeze(0))
                #loss = tp * self.nll_loss(sl.unsqueeze(0), ts.unsqueeze(0))
                loss = tp * tx.losses.sequence_sparse_softmax_cross_entropy(\
                    labels=ts.unsqueeze(0).unsqueeze(0), \
                    logits=sl.unsqueeze(0).unsqueeze(0), \
                    sequence_length=torch.tensor([1]).cuda(), \
                    average_across_timesteps=True, \
                    sum_over_timesteps=False)
                loss_this_sample.append(loss)
            loss_this_sample_data = torch.stack(loss_this_sample)
            losses.append(loss_this_sample_data)
        losses_data = torch.stack(losses)
        avg_loss = losses_data.mean()
        return losses_data, avg_loss

    def _score_w_by_w_on_groundtruth_one_teacher(self, \
            teacher_model, groundtruth, topk_for_memorize_prob=-1, with_softmax=True):
        """
            teacher_model: teacher's nn function
            groundtruth == input_ids (train_batch_size, seq_len)
            if topk_for_memorize_prob == -1; get whole prob
        """
        outputs = teacher_model(inputs=groundtruth, decoding_strategy='train_greedy')
        logits = outputs.logits[:, :-1, :]
        if with_softmax:
            logits = torch.softmax(logits, dim=-1)
        sample_id = outputs.sample_id[:, :-1]
        if -1 == topk_for_memorize_prob:
            vocab_size = -1
            return sample_id, logits, vocab_size
        else:
            t_logits_stopk_v, t_logits_stopk_i, vocab_size, _ = \
                tensor_utils.dense_to_topk_in_sparse(logits, topk_for_memorize_prob, with_softmax=True)
            if self.debug >= 8:
                print("t_logits_stopk_v:", t_logits_stopk_v)
                print("t_logits_stopk_i:", t_logits_stopk_i)
            if self.debug >= 6:
                print("t_logits_stopk_v.shape:", t_logits_stopk_v.shape)
                print("t_logits_stopk_i.shape:", t_logits_stopk_i.shape)
                print("topk_for_memorize_prob:", topk_for_memorize_prob, "vocab_size:", vocab_size)
            return t_logits_stopk_v, t_logits_stopk_i, vocab_size

    #def generate_all_teachermodels(self, testing_file):
    #    results_all = []
    #    for i in range(self.teacher_num):
    #        results_ids, results_ids_list = self.generate(testing_file, self.teachers[i])
    #        results_all.append(results_ids)

    def generate(self, file):
        if self.interactive:
            results_ids, results_ids_list = self.generate_from_prompt(file)
        else:
            results_ids, results_ids_list = self.generate_from_scratch()
        return results_ids, results_ids_list

    def generate_from_prompt(self, file): # For multi-teachers
        # Generate continuations of context
        results_ids_list = []
        results_ids = None

        f = open(file, "r")

        for cnt, line in enumerate(f):
            raw_text = line.strip("\n")
            context_tokens = self.tokenizer.map_text_to_id(raw_text)
            context = torch.tensor(
                [context_tokens for _ in range(self.batch_size)],
                device=self.device)
            context_length = torch.tensor(
                [len(context_tokens) for _ in range(self.batch_size)],
                device=self.device)
            start_tokens = context[:, 0]
            helper = self._get_helper(start_tokens)
    

            logits_all = []
            sample_result_ids_all = []
            for teacher_id in range(self.teacher_num):
                output, _ = self.teachers[teacher_id](
                #output, _ = self.teachers[teacher_id].model(
                    context=context,
                    context_sequence_length=context_length,
                    max_decoding_length=self.max_decoding_length,
                    helper=helper)

                sample_id = output.sample_id
                print("output:", output)
                logits = output.logits
                print("logits.shape:", logits.shape)
                print("logits:", logits)
                logits_all.append(logits)
                sample_result_ids = sample_id[0][len(context_tokens):]
                sample_result_ids_all.append(sample_result_ids)

                #print(self.tokenizer.map_id_to_text(results_ids.tolist()))
                input_results_ids = sample_id[0]
                print("SampleID:\t" + str(cnt) + "\tTeacherID:\t" \
                    + str(teacher_id) + "\t" + \
                    self.tokenizer.map_id_to_text(input_results_ids.tolist()))
                print("Output SampleID:\t" + str(cnt) + "\tTeacherID:\t" \
                    + str(teacher_id) + "\t" + \
                    self.tokenizer.map_id_to_text(input_results_ids[len(context_tokens):].tolist()))
                sys.stdout.flush()

            sample_ids_merge, logits_merge = self.merge_teachers_results(sample_result_ids_all, logits_all)
            print("Merge SampleID:\t" + str(cnt) + "\tTeacherID:\t" \
               + str(teacher_id) + "\t" + \
               self.tokenizer.map_id_to_text(sample_ids_merge[0].tolist()))
            print("")

        results_ids = torch.stack(results_ids_list)
        return results_ids, results_ids_list

    def generate_from_prompt_ori(self, file):
        # Generate continuations of context
        results_ids_list = []
        results_ids = None
        f = open(file, "r")
        for line in f:
            raw_text = line.strip("\n")
            context_tokens = self.tokenizer.map_text_to_id(raw_text)
            context = torch.tensor(
                [context_tokens for _ in range(self.batch_size)],
                device=self.device)
            context_length = torch.tensor(
                [len(context_tokens) for _ in range(batch_size)],
                device=self.device)

            start_tokens = context[:, 0]

            helper = self._get_helper(start_tokens)

            generated = 0
            for _ in range(self.nsamples // self.batch_size):
                output, _ = self.model(
                    context=context,
                    context_sequence_length=context_length,
                    max_decoding_length=self.max_decoding_length,
                    helper=helper)

                sample_id = output.sample_id
                for i in range(self.batch_size):
                    generated += 1
                    print("=" * 40 +
                          " SAMPLE " + str(generated) + " " + "=" * 40)
                    si = sample_id[i][len(context_tokens):]
                    results_ids_list.append(si)
                    print(self.tokenizer.map_id_to_text(si.tolist()))
            print("=" * 80)

        results_ids = torch.stack(results_ids_list)
        return results_ids, results_ids_list

    def generate_from_scratch(self):
        # Generate samples from scratch
        results_ids_list = []
        results_ids = None
        start_tokens = torch.full(
            (self.batch_size,), self.end_token, dtype=torch.int64, device=self.device)

        generated = 0
        while self.nsamples == 0 or generated < self.nsamples:
            helper = self._get_helper(start_tokens)

            output, _ = self.model(
                max_decoding_length=self.max_decoding_length,
                helper=helper)
            sample_id = output.sample_id
            for i in range(self.batch_size):
                generated += self.batch_size
                text = self.tokenizer.map_id_to_text(sample_id[i].tolist())
                print("=" * 40 +
                      " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                results_ids_list.append(sample_id[i])
        results_ids = torch.stack(results_ids_list)

    def merge_teachers_offline(self, t_logits_stopk_v_s, t_logits_stopk_i_s, time_stat=None):
        """
        t_logits_stopk_v_s: list of t_logits_stopk_v 
            [teacher_num, (train_batch, seq_len, topk_vocab)]
        t_logits_stopk_i_s: list of t_logits_stopk_i ()
            [teacher_num, (train_batch, seq_len, topk_vocab)]

        t_logits_dense_merged: (train_batch, seq_len, vocab_size)

        OUTPUT:
        t_logits_merged_stopk_v_s: (train_batch, seq_len, topk_vocab)
        t_logits_merged_stopk_i_s: (train_batch, seq_len, topk_vocab)
        """

        if self.teacher_num <= 0:
            return None

        self.gpu_results_to_cpu_for_memory = False

        t_logits_denses = []
        t_logits_dense_cumulate = None
        t_logits_dense_cumulate_cnt = 0
        new_teacher_id = 0
        for tid in range(self.teacher_num):
            if self.debug >= 7:
                print("len(t_logits_stopk_v_s):", len(t_logits_stopk_v_s), flush=True)
                print("t_logits_stopk_v_s[0].shape:", t_logits_stopk_v_s[0].shape, flush=True)
                print("t_logits_stopk_i_s[0].shape:", t_logits_stopk_i_s[0].shape, flush=True)
            if self.debug >= 10:
                print("t_logits_stopk_v_s[0]:", t_logits_stopk_v_s[0], flush=True)
            #time_stat.begin("get_loss")
            if self.gpu_results_to_cpu_for_memory:
                t_logits_stopk_v_s[tid] = t_logits_stopk_v_s[tid].cuda()
                t_logits_stopk_i_s[tid] = t_logits_stopk_i_s[tid].cuda()

            t_logits_dense = tensor_utils.sparse_topk_to_dense_3D( \
                    t_logits_stopk_v_s[tid], t_logits_stopk_i_s[tid], self.vocab_size, with_softmax=True,\
                    device=self.device, time_stat=time_stat) 
            if self.sparse_merge_to_save_memory_opt == 2:
                t_logits_dense_cumulate, t_logits_dense_cumulate_cnt = \
                    self.tool_online_merge_teacher_prob_or_loss( \
                        t_logits_dense_cumulate, t_logits_dense, t_logits_dense_cumulate_cnt)
                t_logits_dense = None
            else:
                t_logits_denses.append(t_logits_dense)
            #time_stat.end("get_loss")

            # clean current teacher's buf on gpu!!!
            #time_stat.begin("empty_cache")
            torch.cuda.empty_cache()
            #time_stat.end("empty_cache")

            #if tid % 20 == 19:
            if tid % 100 == 99:
                if self.sparse_merge_to_save_memory_opt == 0:
                    t_logits_dense_merged = torch.stack(t_logits_denses).mean(dim=0) 
                elif self.sparse_merge_to_save_memory_opt == 2:
                    t_logits_dense_merged = t_logits_dense_cumulate
                elif self.sparse_merge_to_save_memory_opt == 1:
                    print("len t_logits_denses:", len(t_logits_denses))
                    print("t_logits_denses[0].shape:", t_logits_denses[0].shape)
                    t_logits_dense_merged = self.tool_merge_teacher_prob_or_loss( \
                            t_logits_denses, merge_dim=0, time_stat=time_stat)

                t_logits_merged_stopk_v_s, t_logits_merged_stopk_i_s, vocab_size, _ \
                    = tensor_utils.dense_to_topk_in_sparse(\
                    t_logits_dense_merged, K=100, with_softmax=False)

                ids_for_input, length_for_input = None, None
                self.config_train.teacher_generate_mode = 1
                self.teacher_generate_mode = 1
                self.config_train.save_teacher_results_to_file = 1
                self.save_teacher_results_to_files(ids_for_input, length_for_input, \
                        t_logits_merged_stopk_v_s, t_logits_merged_stopk_i_s, \
                        self.vocab_size, teacher_id=new_teacher_id, is_save_data_when_teacher0=False)
                new_teacher_id += 1

                t_logits_dense_cumulate = None
                t_logits_dense_merged = None
                t_logits_merged_stopk_v_s = None
                t_logits_dense_cumulate = None
                t_logits_dense_cumulate_cnt = 0
                t_logits_merged_stopk_i_s = None
                torch.cuda.empty_cache()

        return None #t_logits_merged_stopk_v_s, t_logits_merged_stopk_i_s
