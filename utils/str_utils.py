"""
Utils of tensor
"""

import texar.torch as tx
import torch
import re

def add_space_around_given_token(str, token):
    """
       xxxxxtokenxxxxx -> xxxxx token xxxxx
    """
    if token not in str:
        return str

    res = str
    lindex = res.index(token)
    if res[lindex-1:lindex] != " ":
        res = res[:lindex] + " " + res[lindex:]

    lindex = res.index(token)
    rindex = lindex + len(token)
    if res[rindex:rindex+1] != " ":
        res = res[:rindex] + " " + res[rindex:]

    return res

def strip_eos(str, eos_token="<EOS>"):
    if eos_token not in str:
        return str
    idx = str.index(eos_token)
    return str[:idx]

def strip_eos_for_list(strs, eos_token="<EOS>"):
    res = []
    for str in strs:
        res.append(strip_eos(str, eos_token))
    return res

def replace_item_for_given_pos(data, pos, replace_data, replace_pos):
    """
       data[:pos] = replace_data[:replace_pos]
    """
    data[pos] = replace_data[replace_pos]
    #for d, rd in zip(data, replace_data):
    #    d[pos] = rd[replace_pos]

def split_word_punctuation_for_list(srcs):
    res = []
    for src in srcs:
        res.append(split_word_punctuation(src))
    return res

def split_word_punctuation(src):
    """
      Hi, Are you OK?  ->  Hi , Are you OK ?
    """
    src_list = re.findall(r"\w+|[^\w\s]", src)
    res = ' '.join(src_list)
    return res

