"""
Utils of tools
"""

import texar.torch as tx
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

def get_train_op(model, config_train):
    if config_train.train_part_of_param_opt == 0: # all params
        train_op = tx.core.get_train_op(
            params=model.parameters(), hparams=config_train.opt)
    elif config_train.train_part_of_param_opt == 1: # only train _output_layer
        train_op = tx.core.get_train_op(
            params=model.decoder._output_layer.parameters(), hparams=config_train.opt)
    elif config_train.train_part_of_param_opt >= 2: 
        # 2: only train _output_layer and poswise_networks
        params = [{"params": model.decoder._output_layer.parameters()}]
        params += [{"params": model.decoder.poswise_networks.parameters()}]

        if config_train.train_part_of_param_opt == 3: 
            # 3: only train _output_layer and poswise_networks and self_attns
            params += [{"params": model.decoder.self_attns.parameters()}]

        train_op = tx.core.get_train_op(
            params=params, hparams=config_train.opt)

    return train_op

def get_curriculum_select_threshold_for_active_learning(config_train, total_batchid):
    if config_train.sac_curriculum_begin == -1:
        return config_train.select_threshold_for_active_learning
    if config_train.sac_curriculum_end == -1:
        return config_train.select_threshold_for_active_learning
    if config_train.sac_curriculum_step == -1:
        return config_train.select_threshold_for_active_learning

    if config_train.sac_curriculum_step <= total_batchid:
        return config_train.sac_curriculum_end

    b = config_train.sac_curriculum_begin
    e = config_train.sac_curriculum_end
    s = config_train.sac_curriculum_step
    old = config_train.select_threshold_for_active_learning

    if b == -1 or e == -1 or s == -1:
        return old

    if s <= total_batchid:
        return e

    res = b + (e - b) / s * total_batchid
    res = int(res)
    print("sac_curriculum b:", b, "e:", e, "s:", s, "old:", old, "total_batchid:", total_batchid, "res:", res)
    return res
