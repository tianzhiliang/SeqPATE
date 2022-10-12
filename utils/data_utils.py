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
"""
Utils of data preprocessing for GPT2 training.
"""

from typing import Any, Dict, List

import os

import texar.torch as tx
import utils.str_utils as str_utils
import random, math

def get_trainfile_list_under_dir(dirname, file_name="train.pkl.pt"):
    res = []
    import os
    for root, dirs, files in os.walk(dirname, topdown=False):
        for name in files:
            if file_name in name:
                full_name = os.path.join(root, name)
                print("file in get_trainfile_list_under_dir:", full_name)
                res.append(full_name)
    return res

def read_raw_data(data_fn: str):
    r"""
    Reads raw data from a file. Each line contains one example.
    """
    examples = []
    with open(data_fn, "r") as fin:
        for line in fin:
            examples.append(line.strip())
    return examples


def convert_examples_to_features_and_output_to_files(
        examples: List[str],
        max_seq_length: int,
        tokenizer: tx.data.GPT2Tokenizer,
        output_file: str,
        feature_types: Dict[str, Any],
        append_eos_token: bool = True):
    r"""Converts a set of examples to a `pickle` file."""

    with tx.data.RecordData.writer(output_file, feature_types) as writer:

        for (_, example) in enumerate(examples):

            text_ids, length = tokenizer.encode_text(
                text=example, max_seq_length=max_seq_length,
                append_eos_token=append_eos_token)

            features = {
                "text_ids": text_ids,
                "length": length
            }
            writer.write(features)  # type: ignore


def prepare_pickle_data(data_dir: str,
                        max_seq_length: int,
                        tokenizer: tx.data.GPT2Tokenizer,
                        output_dir: str,
                        feature_types: Dict[str, Any],
                        is_train_only=False):
    r"""Prepare the `pickle` dataset.
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        tokenizer: The GPT-2 tokenizer.
        output_dir: The directory to save the pickled files in.
        feature_types: The original type of the feature.
    """
    train_fn = os.path.join(data_dir, "train.txt")
    if os.path.isfile(train_fn):
        print("Processing %s" % train_fn)
        train_examples = read_raw_data(train_fn)
        train_file = os.path.join(output_dir, "train.pkl")
        convert_examples_to_features_and_output_to_files(
            train_examples, max_seq_length, tokenizer, train_file,
            feature_types)

    if is_train_only:
        return 

    dev_fn = os.path.join(data_dir, "dev.txt")
    if os.path.isfile(dev_fn):
        print("Processing %s" % dev_fn)
        eval_examples = read_raw_data(dev_fn)
        eval_file = os.path.join(output_dir, "dev.pkl")
        convert_examples_to_features_and_output_to_files(
            eval_examples, max_seq_length, tokenizer, eval_file,
            feature_types)

    test_fn = os.path.join(data_dir, "test.txt")
    if os.path.isfile(test_fn):
        print("Processing %s" % test_fn)
        test_examples = read_raw_data(test_fn)
        test_file = os.path.join(output_dir, "test.pkl")
        convert_examples_to_features_and_output_to_files(
            test_examples, max_seq_length, tokenizer, test_file,
            feature_types, append_eos_token=False)

def load(file):
    f = open(file, "r")
    d = []
    for line in f:
        line = line.strip()
        d.append(line)
    f.close()
    return d

def write_data_to_file(data, file):
    fw = open(file, "w")
    for d in data:
        fw.write(d + "\n")
    fw.close()

def post_process_generated_data(tokenizer, gt_ids, hyp_ids, post_gt_file, post_hyp_file, prefix_length=-1, prompt_len_total_list=None):
    eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')
    print("eos_token_id:(50256?):", eos_token_id)
    
    gts_post=[]
    cnt = 0
    for batch_gt in gt_ids:
        for i in batch_gt:
            print("i:", i)
            print("i.shape:", i.shape)
            i = i.cpu().tolist()
            if i[0]==eos_token_id:
                i = i[1:]
            if prompt_len_total_list is not None and prefix_length == -1:
                real_prefix_length = prompt_len_total_list[cnt] - 1
            else:
                real_prefix_length = prefix_length - 1

            if len(i) <= real_prefix_length:
                i_post = ""
            else:
                i_post = tokenizer.map_id_to_text(i[real_prefix_length:])

            i_post = i_post.replace('\n','').strip()
            gts_post.append(i_post)
            print("i_post:", i_post)

    gts_post = str_utils.strip_eos_for_list(gts_post, eos_token='<|endoftext|>')
    gts_post = str_utils.split_word_punctuation_for_list(gts_post)
    #input_file = os.path.join(args.output_dir,"ori_post.txt")
    print("Write post processed ground truth to {}".format(post_gt_file), flush=True)
    write_data_to_file(gts_post, post_gt_file)

    hyps_post = []
    for cnt_i, s in enumerate(hyp_ids):
        #end_index = s.index(eos_token_id)
        #s = s[:end_index]
        #s_text = tokenizer.map_id_to_text(s)
        #s_text = s_text.replace('\n',' ')
        #s = s.cpu().tolist()
        print("s:", s)
        #print("s.shape:", s.shape)
        #print("s.shape[0]:", s.shape[0])
        #print("prefix_length:", prefix_length)
        try:
            s = s.cpu().tolist()
        except:
            pass

        if prompt_len_total_list is not None and prefix_length == -1:
            real_prefix_length = prompt_len_total_list[cnt_i] - 1
        else:
            real_prefix_length = prefix_length - 1

        if len(s) <= real_prefix_length:
            s_postext = ""
        else:
            s_postext = tokenizer.map_id_to_text(s[real_prefix_length:])

        """if prompt_len_total_list is not None and prefix_length == -1:
            if len(s) <= prompt_len_total_list[cnt_i]:
                s_postext = ""
            else:
                s_postext = tokenizer.map_id_to_text(s[prompt_len_total_list[cnt_i]:])
        else:
            if len(s) <= prefix_length:
                s_postext = ""
            else:
                s_postext = tokenizer.map_id_to_text(s[prefix_length:])
                """

        s_postext = s_postext.replace('\n','').strip()
        print("s_postext:", s_postext)
        # print(s_text,'Post Processed:',s_postext,flush=True)
        hyps_post.append(s_postext)
    hyps_post = str_utils.strip_eos_for_list(hyps_post, eos_token='<|endoftext|>')
    hyps_post = str_utils.split_word_punctuation_for_list(hyps_post)
    print('Write post processed hyp samples (generated samples) to {}'.format(post_hyp_file),flush=True)
    write_data_to_file(hyps_post, post_hyp_file)

def sample_noise_from_vocab_with_cdf(vocab_cdf):
    max_index = len(vocab_cdf) - 1
    rand = random.randint(0, max_index)
    token = vocab_cdf[rand]
    return token

