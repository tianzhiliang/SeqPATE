# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of fine-tuning OpenAI GPT-2 language model.
"""

import argparse
import importlib
import os, sys
from typing import Any
import timeit
import torch
import texar.torch as tx
import utils.time_utils as time_utils
import utils.tools_utils as tools_utils
import my_texar.function_mle_losses as function_mle_losses

import teaching

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pickle_data_dir', type=str, default="None",
    help="pickle_data_dir")
parser.add_argument(
    '--checkpoint', type=str, default="",
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    "--pretrained-model-name", type=str, default="gpt2-small",
    choices=tx.modules.GPT2Decoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    "--trained_teacher_model", type=str, default="model",
    help="Name of trained_teacher_model to load.")
parser.add_argument(
    '--config-train', type=str, default="config_train",
    help="Configurations of GPT-2 training, including data and "
         "optimization hyperparameters.")
parser.add_argument(
    "--output-dir", default="./outputs/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    '--temperature', type=float, default=0.7,
    help="Softmax temperature for top-k sample decoding. Must be strictly "
         "greater than 0. Defaults to 0.7.")
parser.add_argument(
    '--lr', type=float, default=-1,
    help="lr")
parser.add_argument(
    '--pred_output_file', type=str, default='results/result.txt',
    help="Save predicted results")
parser.add_argument(
    '--top-k', type=int, default=40,
    help="The number of top most likely candidates from a vocab distribution.")
parser.add_argument(
    '--teacher_generate_mode', type=int, default=0,
    help="1: teachers generate results and write into files. 0: student training or do not use file to memorize teacher's results ")
parser.add_argument(
    '--top-p', type=float, default=None,
    help="Select tokens with cumulative probability of at most 'p' when "
         "arranged in decreasing order. This will use "
         "TopPSampleEmbeddingHelper for decoding.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")

args = parser.parse_args()
config_train: Any = importlib.import_module(args.config_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
_g_total_step_cnt = 0
_g_total_batchid = 0

def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)
    if args.pickle_data_dir != "None":
        config_train.pickle_data_dir = args.pickle_data_dir
        config_train.train_hparam['dataset']['files'] = args.pickle_data_dir + "/train.pkl"
        config_train.eval_hparam['dataset']['files'] = args.pickle_data_dir + "/dev.pkl"
        config_train.test_hparam['dataset']['files'] = args.pickle_data_dir + "/test.pkl"
        print("args.pickle_data_dir:", args.pickle_data_dir)

    print("args:", args, flush=True)
    print("config_train:", config_train, flush=True)
    print("config_train.__dict__:", config_train.__dict__, flush=True)
    print('Begin training:\n', flush=True)
    max_decoding_length = config_train.max_decoding_length

    # Build the GPT-2 model
    model = tx.modules.GPT2Decoder(args.pretrained_model_name)
    if args.checkpoint != "" and args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)
    model.to(device)
    print("Model:", model)
    if max_decoding_length > model.hparams.position_size:
        raise ValueError(
            "max_decoding_length should not be greater than position size")

    if 0 != config_train.teacher_generate_opt:
        teaching_tool = teaching.Teaching(config_train=config_train, batch_size=config_train.train_batch_size, nsamples=0, \
            max_decoding_length=config_train.max_decoding_length, teacher_model_name_prefix=args.pretrained_model_name, \
            top_p=40, top_k=40, temperature=0.7, interactive=True, \
            device=device, teacher_num=config_train.teacher_num, merge_teachers_opt=2, output_dir=args.output_dir, \
            teacher_generate_mode=args.teacher_generate_mode)

        if args.teacher_generate_mode == 1: # 1: teacher generate dist; 0: student tf
            if config_train.train_batch_size == config_train.teacher_label_batch: # do not label teacher by teacher
                #teaching_tool.init_all_teachermodels(args.trained_teacher_model) # TODO: tmp for debug
                teaching_tool.init_all_teachermodels_to_cpu(args.trained_teacher_model)
            else:
                teaching_tool.init_all_teachermodels_to_cpu(args.trained_teacher_model)
                #teaching_tool.init_for_init_model(args.trained_teacher_model)

        if config_train.save_teacher_results_to_file == 1 and args.teacher_generate_mode == 1: # teacher generate to file
            config_train.max_train_epoch = 1
    else:
        teaching_tool = None

    print('Begin Tokenizing:\n',flush=True)
    # Create a GPT-2 tokenizer (BPE encoding)
    start = timeit.default_timer()
    tokenizer = tx.data.GPT2Tokenizer(
        pretrained_model_name=args.pretrained_model_name)
    stop = timeit.default_timer()
    print('Finish tokenizing, the time cost is:',stop-start,'\n',flush=True)
    print('Begin loading data: \n',flush=True)
    # Loads data
    datasets = {}
    if args.do_train:
        start_time = timeit.default_timer()
        train_dataset = tx.data.RecordData(
            hparams=config_train.train_hparam, device=device)
        stop_time = timeit.default_timer()
        print("Finishing loading data. Time for loading training data:",stop_time-start_time,'\n','-'*100,'\n',flush=True)
        datasets['train'] = train_dataset
    if args.do_eval:
        start = timeit.default_timer()
        eval_dataset = tx.data.RecordData(
            hparams=config_train.eval_hparam, device=device)
        stop = timeit.default_timer()
        print('Finishing loading data. Time for loading validation data:', stop-start,'\n','-'*100,'\n',flush=True)
        datasets['eval'] = eval_dataset
    if args.do_test:
        test_dataset = tx.data.RecordData(
            hparams=config_train.test_hparam, device=device)
        datasets['test'] = test_dataset
    iterator = tx.data.DataIterator(datasets)

    # For training
    if args.lr != -1:
        print("config_train optimizer lr:", config_train.opt['optimizer']['kwargs']['lr'])
        config_train.opt['optimizer']['kwargs']['lr'] = float(args.lr)
        print("config_train optimizer lr:", config_train.opt['optimizer']['kwargs']['lr'])
    #print("config_train optimizer lr:", config_train.opt['optimizer']['lr'])
    if config_train.train_part_of_param_opt == 0: # all params:
        train_op = tx.core.get_train_op(
            params=model.parameters(), hparams=config_train.opt)
    else:
        train_op = tools_utils.get_train_op(model, config_train)

    end_token = tokenizer.map_token_to_id('<|endoftext|>')

    def _get_helper(start_tokens):
        if args.top_p:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                p=args.top_p,
                softmax_temperature=args.temperature)
        else:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                top_k=args.top_k,
                softmax_temperature=args.temperature)
        return helper

    dis_steps = config_train.display_steps
    eval_steps = config_train.eval_steps

    eval_best = {"loss": 1e8, "ppl": 1e8}

    def _student_train_epoch(teaching_tool = None):
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """

        def _get_loss_by_merging_losses(nll_loss, tf_loss):
            global _g_total_step_cnt
            if config_train.curriculum_steps_when_student_topk > 0 and \
                config_train.curriculum_steps_when_student_topk > _g_total_step_cnt:
                tf_loss_weight = 0
            else: # normal branch
                tf_loss_weight = config_train.lambda_soft_hard_teacher_student_loss

            loss = config_train.lambda2_soft_hard_teacher_student_loss * loss_nll \
                + tf_loss * tf_loss_weight

            if config_train.debug >= 4:
                print("nll_loss:", loss_nll.cpu().item(), "tf_loss:", tf_loss.cpu().item(), "loss:", loss.cpu().item())
                if config_train.curriculum_steps_when_student_topk > 0 and config_train.debug >= 10:
                    print("curriculum_steps_when_student_topk:", config_train.curriculum_steps_when_student_topk, \
                            "_g_total_step_cnt:", _g_total_step_cnt)
            if config_train.debug >= 10:
                print("loss:", loss)
            return loss

        iterator.switch_to_dataset("train")
        model.train()

        if config_train.debug >= 5:
            print("step in student_train_epoch")

        step = 0
        wordid2loss, wordid2loss_cnt = {}, {}
        for teacher_label_batchid, batch in enumerate(iterator):
            # Step1: Teacher
            start_time = timeit.default_timer()
            ret = teaching_tool.teacher_collect_or_label(args.trained_teacher_model, \
                batch["text_ids"], batch['length'])
            stop_time = timeit.default_timer()
            print('Time for teacher labeling at each step:',stop_time-start_time,'\n',flush=True)

            if config_train.save_teacher_results_to_file == 1 and args.teacher_generate_mode == 0: # teacher generate to file
                if ret[0] == [None]* 6:
                    print("this epoch done")
                    return 

            config_train.max_train_epoch = 1
            if ret is None: # collecting input data, do not do teacher labeling nor student learning
                print("continue ret:", ret, flush=True)
                continue
            # student learn
            print("ret:", ret, flush=True)

            # Step2: student
            start_time = timeit.default_timer()
            time_stat = time_utils.Time()
            model.to(device)
            for train_batchid, ret_cur in enumerate(ret):
                input_ids, input_lengths, teachers_sample_id, teachers_logits, \
                    t_logits_stopk_v_s, t_logits_stopk_i_s = ret_cur
                start_time2 = timeit.default_timer()

                global _g_total_batchid
                teaching_tool.config_train.select_threshold_for_active_learning = \
                    tools_utils.get_curriculum_select_threshold_for_active_learning(config_train, _g_total_batchid)
                _g_total_batchid += 1

                time_stat.begin("student_model_ff")
                outputs = model(inputs=input_ids, decoding_strategy='train_greedy')

                if config_train.debug >= 7:
                    print("label of loss:", input_ids[:, 1:], "logits of loss:", outputs.logits[:, :-1, :])
                    print("label of loss.shape:", input_ids[:, 1:].shape, "logits of loss.shape:", outputs.logits[:, :-1, :].shape)
                sys.stdout.flush()

                #loss_nll = function_mle_losses.my_sequence_sparse_softmax_cross_entropy(
                #loss_nll = function_mle_losses.my_sequence_sparse_softmax_cross_entropy_prob_topk(
                loss_nll = tx.losses.sequence_sparse_softmax_cross_entropy(
                    labels=input_ids[:, 1:],
                    logits=outputs.logits[:, :-1, :],
                    sequence_length=input_lengths - 1,
                    average_across_timesteps=True,
                    sum_over_timesteps=False)
                    #device=device,
                    #loss_valid_prob_topk=config_train.loss_valid_topk)
                #function_mle_losses.print_cumulate_loss_on_wordid(wordid2loss, wordid2loss_cnt, "tmp.txt")
                time_stat.end("student_model_ff")

                time_stat.begin("tf_loss_in_main")
                print("gpt2 time_stat:", time_stat)
                if config_train.teacher_generate_opt is not None and 0 != config_train.teacher_generate_opt and teaching_tool is not None:
                    if config_train.topk_for_memorize_prob <= 0:
                        tf_loss = teaching_tool.get_ts_loss_wo_ff(outputs, teachers_sample_id, teachers_logits, labels_for_stat=input_ids[:, 1:], time_stat=time_stat)
                    else:
                        tf_loss = teaching_tool.get_ts_loss_with_topk_vocab(input_ids, outputs, \
                                train_batchid, t_logits_stopk_v_s, t_logits_stopk_i_s, labels_for_stat=input_ids[:,1:], time_stat=time_stat, step=step)

                    loss = _get_loss_by_merging_losses(loss_nll, tf_loss)
                    #loss = config_train.lambda2_soft_hard_teacher_student_loss * loss_nll \
                    #    + tf_loss * config_train.lambda_soft_hard_teacher_student_loss
                    #if config_train.debug >= 4:
                    #    print("nll_loss:", loss_nll.cpu().item(), "tf_loss:", tf_loss.cpu().item(), "loss:", loss.cpu().item())
                    #if config_train.debug >= 10:
                    #    print("loss:", loss)
                else:
                    loss = loss_nll

                time_stat.end("tf_loss_in_main")
                loss.backward()
                #loss.backward(retain_graph=True)
                train_op()

                torch.cuda.empty_cache()

                if dis_steps > 0 and step % dis_steps == 0:
                    print("step={}, loss={:.4f}".format(step, loss),flush=True)

                stop_time2 = timeit.default_timer()
                print('Time for each step:',stop_time2-start_time2,'\n',flush=True)

            model.to(cpu_device)
            torch.cuda.empty_cache()
            if config_train.debug >= 5:
                time_stat.print_all()
            stop_time = timeit.default_timer()
            print('Time for training the student on teacher\'s label:', stop_time-start_time,'\n',flush=True)

            if step!=0 and eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()
                model.train()
            step += 1
            global _g_total_step_cnt 
            _g_total_step_cnt += 1

        _eval_epoch()
        model.train()

    def _train_epoch(teaching_tool = None):
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()

        step = 0
        for batch in iterator:
            start_time = timeit.default_timer()
            input_ids = batch["text_ids"]

            outputs = model(inputs=input_ids, decoding_strategy='train_greedy')

            if config_train.debug >= 7:
                print("label of loss:", batch['text_ids'][:, 1:], "logits of loss:", outputs.logits[:, :-1, :])
                print("label of loss.shape:", batch['text_ids'][:, 1:].shape, "logits of loss.shape:", outputs.logits[:, :-1, :].shape)

            loss_nll = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)

            if config_train.teacher_generate_opt is not None and 0 != config_train.teacher_generate_opt and teaching_tool is not None:
                tf_loss = teaching_tool.get_ts_loss(input_ids, outputs)
                loss = config_train.lambda2_soft_hard_teacher_student_loss * loss_nll \
                    + tf_loss * config_train.lambda_soft_hard_teacher_student_loss
                if config_train.debug >= 4:
                    print("nll_loss:", loss_nll.cpu().item(), "tf_loss:", tf_loss.cpu().item(), "loss:", loss.cpu().item())
            else:
                loss = loss_nll
            loss.backward()
            train_op()

            if dis_steps > 0 and step % dis_steps == 0:
                print("step={}, loss={:.4f}".format(step, loss),flush=True)

            stop_time = timeit.default_timer()
            print('Time for each step:',stop_time-start_time,'\n',flush=True)
            if step!=0 and eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()
                model.train()
            step += 1

        _eval_epoch()
        model.train()

    @torch.no_grad()
    def _eval_epoch():
        r"""Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()

        nsamples = 0
        wordid2loss, wordid2loss_cnt = {}, {}
        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:
            input_ids = batch["text_ids"]

            #print("input_ids:", input_ids)
            #print("model:", model)
            model.to(device)
            #print("after to device model:", model)
            outputs = model(inputs=input_ids)

            #loss = function_mle_losses.my_sequence_sparse_softmax_cross_entropy_prob_topk(
            #loss = function_mle_losses.my_sequence_sparse_softmax_cross_entropy(
            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)
                #device=device,
                #loss_valid_prob_topk=config_train.loss_valid_topk)
                #wordid2loss=wordid2loss,
                #wordid2loss_cnt=wordid2loss_cnt)
            ppl = torch.exp(loss)
            batch_size = input_ids.size()[0]
            avg_rec.add([loss, ppl], batch_size)
            nsamples += batch_size

        print("eval loss: {:.4f}; ppl: {:.4f}; "
              "nsamples: {:d}".format(avg_rec.avg(0), avg_rec.avg(1), nsamples),flush=True)
        #function_mle_losses.print_cumulate_loss_on_wordid(wordid2loss, wordid2loss_cnt, "tmp.txt")

        if args.do_train and avg_rec.avg(0) < eval_best["loss"]:
            eval_best["loss"] = avg_rec.avg(0)
            eval_best["ppl"] = avg_rec.avg(1)
            ckpt_fn = os.path.join(args.output_dir, 'model_best.ckpt')
            torch.save(model.state_dict(), ckpt_fn)
            print("Checkpoint best to {}".format(ckpt_fn))

    @torch.no_grad()
    def _test_epoch():
        r"""Generates samples on the test set.
        """
        iterator.switch_to_dataset("test")
        model.eval()

        _all_inputs = []
        _all_samples = []

        for batch in iterator:
            input_ids = batch["text_ids"]
            length = batch["length"]
            start_tokens = input_ids[:, 0]
            helper = _get_helper(start_tokens)

            output, _ = model(
                context=input_ids,
                context_sequence_length=length,
                max_decoding_length=max_decoding_length,
                helper=helper)
            sample_id = output.sample_id

            _inputs = []
            for i, l in zip(input_ids, length):
                # Delete padding
                _inputs.append(i[:l].tolist())
            _all_inputs.extend(_inputs)

            _samples = []
            for s, l in zip(sample_id, length):
                # Delte inputs from samples
                _samples.append(s[l:].tolist())
            _all_samples.extend(_samples)

        # Parse samples and write to file

        eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                # '<|endoftext|>' is used as the BOS token. Delete it here
                i = i[1:]
            i_text = tokenizer.map_id_to_text(i)
            _all_input_text.append(i_text)
        # '<|endoftext|>' is used as the PAD token. Delete them here
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = tokenizer.map_id_to_text(s)
            s_text = s_text.replace('\n', ' ')
            _all_samples_text.append(s_text)
        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                               eos_token='<|endoftext|>')

        output_file = os.path.join(args.output_dir, "test_samples.tsv")
        print('Write samples to {}'.format(output_file),flush=True)
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)

    if args.do_train:
        for _ in range(config_train.max_train_epoch):
            if teaching_tool is None:
            #if teaching_tool is None or config_train.train_batch_size == config_train.teacher_label_batch:
                # do not label teacher by teacher or normal training (without teacher labeling)
                _train_epoch(teaching_tool)
            elif teaching_tool is not None: # label teacher by teacher
                _student_train_epoch(teaching_tool)
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
