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

import torch
import texar.torch as tx
import utils.tensor_utils as tensor_utils
import utils.str_utils as str_utils
import utils.data_utils as data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    "--pretrained-model-name", type=str, default="gpt2-small",
    choices=tx.modules.GPT2Decoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    '--config-train', type=str, default="config_train",
    help="Configurations of GPT-2 training, including data and "
         "optimization hyperparameters.")
parser.add_argument(
    "--output-dir", default="./outputs/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    '--pred_output_file', type=str, default='results/result.txt',
    help="Save predicted results")
parser.add_argument(
    '--temperature', type=float, default=0.7,
    help="Softmax temperature for top-k sample decoding. Must be strictly "
         "greater than 0. Defaults to 0.7.")
parser.add_argument(
    '--top-k', type=int, default=40,
    help="The number of top most likely candidates from a vocab distribution.")
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
parser.add_argument(
    "--teacher_generate_opt", type=int, default=1,
    help="0: generate like teacher-forcing; 1:generate as inference")
parser.add_argument(
    "--prompt_token_num", type=int, default=-1,
    help="-1 means all tokens")
parser.add_argument(
    "--is_splited_prefix_file", type=int, default=0,
    help="1: input file is the prefix (splited by string offline; this program does not split and generate gt file; 0: full original input file (this program split prefix and obtain gt))")

args = parser.parse_args()

config_train: Any = importlib.import_module(args.config_train)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompt_token_num_with_start = args.prompt_token_num + 1

def remove_from_list(src, value):
    while True:
        try:
            src.remove(value)
        except:
            return src

def replace_item_in_list(l, src, tgt):
    lens = len(l)
    for i in range(lens):
        if l[i] == src:
            l[i] = tgt
    return l

def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    max_decoding_length = config_train.max_decoding_length

    # Build the GPT-2 model
    model = tx.modules.GPT2Decoder(args.pretrained_model_name)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)
    model.to(device)

    if max_decoding_length > model.hparams.position_size:
        raise ValueError(
            "max_decoding_length should not be greater than position size")

    # Create a GPT-2 tokenizer (BPE encoding)
    tokenizer = tx.data.GPT2Tokenizer(
        pretrained_model_name=args.pretrained_model_name)

    # Loads data
    datasets = {}
    if args.do_train:
        train_dataset = tx.data.RecordData(
            hparams=config_train.train_hparam, device=device)
        datasets['train'] = train_dataset
    if args.do_eval:
        eval_dataset = tx.data.RecordData(
            hparams=config_train.eval_hparam, device=device)
        datasets['eval'] = eval_dataset
    if args.do_test:
        test_dataset = tx.data.RecordData(
            hparams=config_train.test_hparam, device=device)
        datasets['test'] = test_dataset
    iterator = tx.data.DataIterator(datasets)

    # For training
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_train.opt)

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

    def _train_epoch():
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()

        step = 0
        for batch in iterator:
            input_ids = batch["text_ids"]

            outputs = model(inputs=input_ids, decoding_strategy='train_greedy')

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)
            loss.backward()
            train_op()

            if dis_steps > 0 and step % dis_steps == 0:
                print("step={}, loss={:.4f}".format(step, loss))

            if eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()
                model.train()

            step += 1

    @torch.no_grad()
    def _eval_epoch():
        r"""Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:
            input_ids = batch["text_ids"]

            outputs = model(inputs=input_ids)

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)
            ppl = torch.exp(loss)
            batch_size = input_ids.size()[0]
            avg_rec.add([loss, ppl], batch_size)
            nsamples += batch_size

        print("eval loss: {:.4f}; ppl: {:.4f}; "
              "nsamples: {:d}".format(avg_rec.avg(0), avg_rec.avg(1), nsamples))

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

        _all_input_ids = []
        _all_samples = []

        avg_rec = tx.utils.AverageRecorder()
        nsamples=0
        loss = None

        output_file = os.path.join(args.output_dir, "test_samples.txt")
        print('Write samples to {}'.format(output_file),flush=True)
       #  tx.utils.write_paired_text(
       #      _all_input_text, _all_samples_text, output_file)
        fw = open(output_file, 'w')
        prompt_len_total_list = []

        for batch_cnt, batch in enumerate(iterator):
            input_ids = batch["text_ids"]
            _all_input_ids.append(input_ids)
            # The first token is the start token. Different from the generate_main.py
            batch_size=input_ids.size()[0]
            # length = batch["length"]
            if config_train.teacher_generate_opt == 1: 
                #print('teacher_generate_opt==1. input_ids.size():',input_ids.size(), "input_ids:", input_ids)
                sys.stdout.flush()
                start_tokens=input_ids[:,0]
                helper = _get_helper(start_tokens)
                if 0 == prompt_token_num_with_start: # only support batchsize=1
                    eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')
                    cxt_length = torch.tensor([0]*batch_size,device=device)
                    prompt_list = []
                    for i in range(input_ids.size()[0]):
                        print("input_ids[i]:", input_ids[i])
                        try:
                            end_index = input_ids[i][1:].cpu().tolist().index(eos_token_id) + 1
                        except:
                            end_index = int(input_ids[i].size()[0])
                        print("in prompt_token deal end_index:", end_index)
                        prompt_len_total_list.append(end_index)
                        prompt_each = torch.tensor(input_ids[i][:end_index].cpu().tolist() + \
                            [eos_token_id] * (input_ids[i].size()[0] - end_index), device=device)
                        prompt_list.append(prompt_each)
                        cxt_length[i] = end_index
                    prompt = torch.stack(prompt_list)
                else:
                    prompt = torch.tensor(input_ids[:,:prompt_token_num_with_start],device=device)# Since it contains the start token, therefore, we choose the first four tokens as prompt.
                    cxt_length = torch.tensor([prompt_token_num_with_start]*batch_size,device=device)

                output, _ = model(
                    context=prompt,
                    context_sequence_length=cxt_length,
                    max_decoding_length=max_decoding_length,
                    helper=helper)

                #if 0 != prompt_token_num_with_start:
                    # max_time = min(output.logits.size()[1]-1,input_ids.size()[1])# Make sure the max time of labels and generated sentence are equal.
                    #max_time = output.logits.size()[1]-1
                    #if max_time > prompt_token_num_with_start:
                    #    seq_length = torch.tensor([max_time]*batch_size, device=device)
            # Compute loss and perplexity:
                    #    loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                    #        labels=batch['text_ids'][:,prompt_token_num_with_start:max_time],# We only consider the aligned tokens we may generate.  
                    #        logits=output.logits[:,prompt_token_num_with_start:max_time,:],
                    #        sequence_length=seq_length-1,# sequence length function, ignore the [EOS].
                    #        average_across_timesteps=True,
                    #        sum_over_timesteps=False)

                print("output.sample_id:", output.sample_id)
                samples_id = output.sample_id[:, 1:] # rm first token (<|endoftext|>)
                print("sample_id:", samples_id)
                for sid, sample_id in enumerate(samples_id):
                    sample_id_list = sample_id.cpu().tolist()
                    eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')
                    sample_id_list = replace_item_in_list(sample_id_list, 198, eos_token_id) # 198 may be '\n'
                    sample_id_list = replace_item_in_list(sample_id_list, 628, eos_token_id) # 198 may be '\n'
                    #empty_id = tokenizer.map_token_to_id(' ')
                    #enter_ch = tokenizer.map_id_to_token(198) # 198 may be '\n'
                    #print("empty_id:", empty_id, "enter_ch:", enter_ch)
                    sample_id_list += [eos_token_id]
                    #sample_id_list = remove_from_list(sample_id_list, 198) # 198 may be '\n'
                    end_index = sample_id_list.index(eos_token_id)
                    print("ori sample_id_list:", sample_id_list, "eos_token_id:", eos_token_id, "end_index:", end_index)
                    sample_id_list = sample_id_list[:end_index]
                    _all_samples.append(sample_id_list)
                    sample_str = tokenizer.map_id_to_text(sample_id_list)
                    #sample_str = sample_str.replace(enter_ch, ' aaaaaa ')
                    print("sample_cnt:", sid, "nsamples:", nsamples+sid, "sample_str:", sample_str)
                    fw.write(sample_str + "\n")
                fw.flush()
            elif config_train.teacher_generate_opt == 0:
                #input_ids_wo_first = input_ids[:, 1:]
                prompt = torch.tensor(input_ids[:,:prompt_token_num_with_start],device=device)# Since it contains the start token, therefore, we choose the first four tokens as prompt.
                cxt_length = torch.tensor([prompt_token_num_with_start]*batch_size,device=device)
                outputs = model(inputs=input_ids, \
                    context=prompt, \
                    context_sequence_length=cxt_length, \
                    decoding_strategy='train_greedy')
                #outputs = model(inputs=input_ids, decoding_strategy='train_greedy')

                labels = batch['text_ids'][:, 1:]
                loss = None
                #loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                #    labels=labels,
                #    logits=outputs.logits[:, :-1, :],
                #    sequence_length=batch['length'] - 1,
                #    average_across_timesteps=True,
                #    sum_over_timesteps=False)
                sample_ids = outputs.sample_id
                eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')
                sample_ids = tensor_utils.truncate_as_given_shape(sample_ids, labels, eos_token_id)
                for sid, sample in enumerate(sample_ids):
                    #sample = sample.cpu()
                    #print("sample:", sample, "eos_token_id:", eos_token_id)
                    #print("vocab encoder:", tokenizer.encoder)
                    #print("vocab decoder:", tokenizer.decoder)
                    #print("input_ids:", input_ids)
                    #print("sample map_id_to_text:", tokenizer.map_id_to_token(sample.tolist()))
                    sample = sample.cpu().tolist()
                    str_utils.replace_item_for_given_pos(sample, 0, input_ids[sid].cpu().tolist(), 1)
                    sample_str = tokenizer.map_id_to_text(sample)
                    sample_str = str_utils.add_space_around_given_token(sample_str, '<|endoftext|>')
                    sample_str = sample_str.replace('\n','')
                    print("sample_cnt:", sid, "nsamples:", nsamples+sid, "sample_str:", sample_str)
                    fw.write(sample_str + "\n")
                _all_samples += sample_ids 
                print("batch_cnt:", batch_cnt, "nsamples:", nsamples)
                fw.flush()

            if loss is not None:
                ppl=torch.exp(loss)
                print('Perplexity for this sample is %.4f'%ppl,flush=True)
                avg_rec.add([loss,ppl],batch_size)
            nsamples += batch_size

        if args.is_splited_prefix_file == 0:
            data_utils.post_process_generated_data(tokenizer, _all_input_ids, \
                _all_samples, output_file + ".gt", output_file + ".hyp", \
                prefix_length=prompt_token_num_with_start)
        elif args.is_splited_prefix_file == 1:
            data_utils.post_process_generated_data(tokenizer, _all_input_ids, \
                _all_samples, output_file + ".gt", output_file + ".hyp", \
                prompt_len_total_list=prompt_len_total_list)


    if args.do_train:
        for _ in range(config_train.max_train_epoch):
            _train_epoch()
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
