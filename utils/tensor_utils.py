"""
Utils of tensor
"""

import texar.torch as tx
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

def tensor_equal(a, b):
    return ((torch.eq(a,b)*1.0).mean().cpu().item() == 1.0)

def set_value_where_src_eq_value(src, ori_dest, value=0):
    """
      if src[i] == value; set dest[i] = value
      else: dest[i] = dest[i] (do not change)
    """

    diff = value # diff = value - 0
    dest = ori_dest.clone()
    target_slots = src != value #!= is 1; == is 0
    dest_s_diff = dest - diff
    dest = dest_s_diff * target_slots
    dest = dest + diff
    #res_dest = res_dest + diff
    return dest

def set_value_where_src_eq_value_wo_clone(src, dest, value=0):
    """
      if src[i] == value; set dest[i] = value
      else: dest[i] = dest[i] (do not change)
    """

    diff = value # diff = value - 0
    target_slots = src != value #!= is 1; == is 0
    dest_s_diff = dest - diff
    dest = dest_s_diff * target_slots
    dest = dest + diff
    return dest

def set_valueA_where_src_smaller_valueB(src, dest, valueA, valueB, time_stat=None):
    """
      if src[i] <= valueA; set dest[i] = valueB
    """

    #time_stat.begin("set_valueA_where_src_eq_valueB_0_assign")
    diff = valueB # diff = value - 0
    #time_stat.end("set_valueA_where_src_eq_valueB_0_assign")
    #time_stat.begin("set_valueA_where_src_eq_valueB_1_clone")
    #time_stat.end("set_valueA_where_src_eq_valueB_1_clone")
    #time_stat.begin("set_valueA_where_src_eq_valueB_2_not_equal")
    target_slots = src > valueA #!= is 1; == is 0
    #time_stat.end("set_valueA_where_src_eq_valueB_2_not_equal")
    #time_stat.begin("set_valueA_where_src_eq_valueB_3_subscibe")
    #print("src:", src)
    #print("valueA:", valueA)
    #print("target_slots:", target_slots)
    dest_s_diff = dest - diff
    #time_stat.end("set_valueA_where_src_eq_valueB_3_subscibe")
    #time_stat.begin("set_valueA_where_src_eq_valueB_4_mul")
    dest = dest_s_diff * target_slots
    #time_stat.end("set_valueA_where_src_eq_valueB_4_mul")
    #time_stat.begin("set_valueA_where_src_eq_valueB_5_add")
    #print("dest:", dest)
    dest = dest + diff
    #time_stat.end("set_valueA_where_src_eq_valueB_5_add")
    #print("diff:", diff)
    #print("dest+diff:", dest)
    #res_dest = res_dest + diff
    return dest

def set_valueA_where_src_smaller_valueB_with_clone(src, ori_dest, valueA, valueB, time_stat=None):
    """
      if src[i] <= valueA; set dest[i] = valueB
    """

    #time_stat.begin("set_valueA_where_src_eq_valueB_0_assign")
    diff = valueB # diff = value - 0
    #time_stat.end("set_valueA_where_src_eq_valueB_0_assign")
    #time_stat.begin("set_valueA_where_src_eq_valueB_1_clone")
    dest = ori_dest.clone()
    #time_stat.end("set_valueA_where_src_eq_valueB_1_clone")
    #time_stat.begin("set_valueA_where_src_eq_valueB_2_not_equal")
    target_slots = src > valueA #!= is 1; == is 0
    #time_stat.end("set_valueA_where_src_eq_valueB_2_not_equal")
    #time_stat.begin("set_valueA_where_src_eq_valueB_3_subscibe")
    #print("src:", src)
    #print("valueA:", valueA)
    #print("target_slots:", target_slots)
    dest_s_diff = dest - diff
    #time_stat.end("set_valueA_where_src_eq_valueB_3_subscibe")
    #time_stat.begin("set_valueA_where_src_eq_valueB_4_mul")
    dest = dest_s_diff * target_slots
    #time_stat.end("set_valueA_where_src_eq_valueB_4_mul")
    #time_stat.begin("set_valueA_where_src_eq_valueB_5_add")
    #print("dest:", dest)
    dest = dest + diff
    #time_stat.end("set_valueA_where_src_eq_valueB_5_add")
    #print("diff:", diff)
    #print("dest+diff:", dest)
    #res_dest = res_dest + diff
    return dest

def set_valueA_where_src_larger_valueB(src, ori_dest, valueA, valueB, time_stat=None):
    """
      if src[i] >= valueA; set dest[i] = valueB
    """

    #time_stat.begin("set_valueA_where_src_eq_valueB_0_assign")
    diff = valueB # diff = value - 0
    #time_stat.end("set_valueA_where_src_eq_valueB_0_assign")
    #time_stat.begin("set_valueA_where_src_eq_valueB_1_clone")
    dest = ori_dest.clone()
    #time_stat.end("set_valueA_where_src_eq_valueB_1_clone")
    #time_stat.begin("set_valueA_where_src_eq_valueB_2_not_equal")
    target_slots = src < valueA #!= is 1; == is 0
    #time_stat.end("set_valueA_where_src_eq_valueB_2_not_equal")
    #time_stat.begin("set_valueA_where_src_eq_valueB_3_subscibe")
    #print("src:", src)
    #print("valueA:", valueA)
    #print("target_slots:", target_slots)
    dest_s_diff = dest - diff
    #time_stat.end("set_valueA_where_src_eq_valueB_3_subscibe")
    #time_stat.begin("set_valueA_where_src_eq_valueB_4_mul")
    dest = dest_s_diff * target_slots
    #time_stat.end("set_valueA_where_src_eq_valueB_4_mul")
    #time_stat.begin("set_valueA_where_src_eq_valueB_5_add")
    #print("dest:", dest)
    dest = dest + diff
    #time_stat.end("set_valueA_where_src_eq_valueB_5_add")
    #print("diff:", diff)
    #print("dest+diff:", dest)
    #res_dest = res_dest + diff
    return dest

def set_valueA_where_src_eq_valueB(src, ori_dest, valueA, valueB, time_stat):
    """
      if src[i] == valueA; set dest[i] = valueB
    """

    #time_stat.begin("set_valueA_where_src_eq_valueB_0_assign")
    diff = valueB # diff = value - 0
    #time_stat.end("set_valueA_where_src_eq_valueB_0_assign")
    #time_stat.begin("set_valueA_where_src_eq_valueB_1_clone")
    dest = ori_dest.clone()
    #time_stat.end("set_valueA_where_src_eq_valueB_1_clone")
    #time_stat.begin("set_valueA_where_src_eq_valueB_2_not_equal")
    target_slots = src != valueA #!= is 1; == is 0
    #time_stat.end("set_valueA_where_src_eq_valueB_2_not_equal")
    #time_stat.begin("set_valueA_where_src_eq_valueB_3_subscibe")
    #print("src:", src)
    #print("valueA:", valueA)
    #print("target_slots:", target_slots)
    dest_s_diff = dest - diff
    #time_stat.end("set_valueA_where_src_eq_valueB_3_subscibe")
    #time_stat.begin("set_valueA_where_src_eq_valueB_4_mul")
    dest = dest_s_diff * target_slots
    #time_stat.end("set_valueA_where_src_eq_valueB_4_mul")
    #time_stat.begin("set_valueA_where_src_eq_valueB_5_add")
    #print("dest:", dest)
    dest = dest + diff
    #time_stat.end("set_valueA_where_src_eq_valueB_5_add")
    #print("diff:", diff)
    #print("dest+diff:", dest)
    #res_dest = res_dest + diff
    return dest

def assign_tensorA_element_to_tensorB_element_where_masked( \
        tensorA, tensorB, mask, time_stat=None, \
        avoid_negative_value=False, device=torch.device("cpu")):
    """
    Input:
      tensorA: ori src (batch, seqlen, vocabsize) (e.g. teacher's prob)
               static, assign A's value to B's
      tensorB: dest (batch, seqlen, vocabsize) (e.g. student's prob)
               will be wrttien by A where masked
      mask: binary matrix (batch, seqlen, vocabsize) (e.g. non-topk matrix)
            0: non masked  1: masked
      avoid_negative_value: whether avoid negative value in the results
            due to the float precision, the results may be a little larger/small than the original value
            if the results are probability, negative value may lead to nan in following layers
            so, we should avoid negative value

    Output:
      tensorB: dest (batch, seqlen, vocabsize) (e.g. student's prob)
               will be wrttien by A where masked

    Operateion:
      if mask[i] == 1; B[i] = A[i]
      if mask[i] == 0; B[i] = B[i] (unchanged)
      ignore the gradient (in my application, A is teacher (detached already); 
            B is student (if assigned, then B=A, then no loss, then no grad to BP))
    """

    # mask: non-topK value is 1, topK value is 0
    # reversed_mark: non-topK value is 0, topK value is 1
    reversed_mark = 1.0 - mask

    # non-topk's value is 0
    tensorB_topk_only = tensorB * reversed_mark

    tensorA_nontopk_only = tensorA * mask

    tensorB_v2 = tensorB_topk_only + tensorA_nontopk_only

    if avoid_negative_value:
        tensorB_v2 = torch.abs(tensorB_v2)

    almost_zero = torch.tensor(1e-15, device=device, requires_grad=False)
    tensorB_v3 = set_valueA_where_src_smaller_valueB(\
        tensorB_v2, tensorB_v2, almost_zero, almost_zero)

    return tensorB_v3

def truncate_as_given_shape(input, shape_data, eos_id):
    """
        output has the same shape with "shape_data" (shape_data use eos_id to mark the empty token)
        the output may not be back-propagate? 
    """
    outputs = []
    for i, sd in enumerate(shape_data):
        #print("sd.size():", sd.size())
        #print("input.shape:", input.shape)
        #print("input[i].shape:", input[i].shape)
        #print("sd.size()[0]:", sd.size()[0])
        #print("sd:", sd)
        try:
            end_index = sd.cpu().tolist().index(eos_id)
            outputs.append(input[i][:end_index])
        except:
            outputs.append(input[i])
        #end_index2 = input[i].cpu().tolist().index(50256)
        #print("end_index:", end_index)
        #print("end_index:", end_index, "end_index2:", end_index2)
        #print("sd:", sd, "input[i]:", input[i])
    #output_data = torch.stack(outputs)
    #print("output_data.shape:", output_data.shape)
    #return output_data
    return outputs

def dense_to_topk_in_sparse_2D(dense, K=100):
    """
    dense: source data e.g. (batch, vocabsize)
    K: topk
    dim_of_topk = -1: can only select topk from the last dim

    output:
        topk_value: topk value e.g. (batch, K)
        topk_idx: topk index e.g. (batch, K)
        dense_ori_dim: vocabsize
    """

    dim_of_topk = -1
    dense_ori_dim = dense.shape[dim_of_topk]
    dense_v1 = dense.reshape(-1, dense_ori_dim)
    topk_value_v1, topk_idx_v1 = torch.topk(dense_v1, K, dim=dim_of_topk)

    topk_value = topk_value_v1.reshape(list(dense.shape[:-1]) + [K])
    topk_idx = topk_idx_v1.reshape(list(dense.shape[:-1]) + [K])

    return topk_value, topk_idx, dense_ori_dim

def dense_to_topk_in_sparse(dense, K=100, with_softmax = True):
    """
    dense: source data e.g. (batch, seqlen, vocabsize)
    K: topk
    dim_of_topk = -1: can only select topk from the last dim

    output:
        topk_value: topk value e.g. (batch, seqlen, K)
        topk_idx: topk index e.g. (batch, seqlen, K)
        dense_ori_dim: vocabsize
    """

    dim_of_topk = -1
    dense_ori_dim = dense.shape[dim_of_topk]
    dense_v1 = dense.reshape(-1, dense_ori_dim)
    topk_value_v1, topk_idx_v1 = torch.topk(dense_v1, K, dim=dim_of_topk)

    topk_value = topk_value_v1.reshape(list(dense.shape[:-1]) + [K])
    topk_idx = topk_idx_v1.reshape(list(dense.shape[:-1]) + [K])

    sum_of_all_values = None #(if with_softmax: sum_of_all_values=1)
    if not with_softmax:
        print("dense_to_topk_in_sparse not with_softmax")
        sum_of_all_values = dense.sum(dim=dim_of_topk)

    return topk_value, topk_idx, dense_ori_dim, sum_of_all_values

def sparse_topk_to_dense_2D(sparse_topk_value, sparse_topk_idx, dense_ori_dim, \
     with_softmax=True, device=None, time_stat=None, sum_of_all_values=None, remainder_is_0=False):
    """
    sparse_topk_value: (batchsize, K) topk_value from dense_to_topk_in_sparse
    sparse_topk_idx: (batchsize, K) topk_idx from dense_to_topk_in_sparse
    dense_ori_dim: 1 (the original dim of the original dense matrix)

    input must be normalized to 0~1; otherwise, cannot use sum_of_topk to obtain the remainder
    """

    dim_of_non_topk = 0
    dim_of_topk = -1
    K = sparse_topk_value.shape[dim_of_topk]

    orishape = sparse_topk_value.shape
    if time_stat is not None:
        time_stat.begin("sparse_topk_to_dense_2D_fillin")

    # Step1: init as all (1-sum(prob_of_topk))/ (N-topk)
    # sum_of_topk: (batchsize)
    # remainder: (batchsize)
    if time_stat is not None:
        time_stat.begin("sparse_topk_to_dense_2D_remainder")
    sum_of_topk = sparse_topk_value.sum(dim=dim_of_topk)
    if with_softmax:
        sum_of_total = 1
        remainder = (sum_of_total-sum_of_topk) * 1.0 / (dense_ori_dim - K)
    elif not remainder_is_0:
        print("sum_of_all_values:", sum_of_all_values)
        print("sum_of_all_values.shape:", sum_of_all_values.shape)
        print("sum_of_topk.shape:", sum_of_topk.shape)
        sum_of_total = sum_of_all_values
        remainder = (sum_of_total-sum_of_topk) * 1.0 / (dense_ori_dim - K)
    elif remainder_is_0:
        remainder = sum_of_topk * 0.0
        print("in remainder_is_0")
        print("remainder:", remainder, "remainder.shape:", remainder.shape)

    #remainder = Variable(torch.tensor(remainder).float())
    remainder = Variable(remainder.clone().detach().float())
    if time_stat is not None:
        time_stat.end("sparse_topk_to_dense_2D_remainder")
        time_stat.begin("sparse_topk_to_dense_2D_remainder_tocuda")
    remainder.to(device)
    if time_stat is not None:
        time_stat.end("sparse_topk_to_dense_2D_remainder_tocuda")
    #print("remainder:", remainder)
    #print("dense_ori_dim:", dense_ori_dim)

    # Step2
    denses = []
    for i in range(orishape[0]):
        if time_stat is not None:
            time_stat.begin("sparse_topk_to_dense_2D_fillin_step")
        index = sparse_topk_idx[i]
        #print("index:", index)
        index = index.unsqueeze(0)
        value = sparse_topk_value[i]
        #print("index:", index)
        #print("value:", value)
        if time_stat is not None:
            time_stat.begin("sparse_topk_to_dense_2D_fillin sparse")
        sparse_v1 = torch.sparse.FloatTensor(index, value, torch.Size([dense_ori_dim]))

        if time_stat is not None:
            time_stat.end("sparse_topk_to_dense_2D_fillin sparse")
            time_stat.begin("sparse_topk_to_dense_2D_fillin sparse_to_dense")
        dense = sparse_v1.to_dense()

        if time_stat is not None:
            time_stat.end("sparse_topk_to_dense_2D_fillin sparse_to_dense")
            time_stat.begin("sparse_topk_to_dense_2D_fillin add remainder")
        #print("dense:", dense)

        dense = set_valueA_where_src_eq_valueB(dense, dense, 0.0, remainder[i], time_stat=time_stat)
        #print("dense after fillin:", dense)
        denses.append(dense)
        if time_stat is not None:
            time_stat.end("sparse_topk_to_dense_2D_fillin add remainder")
            time_stat.end("sparse_topk_to_dense_2D_fillin_step")

    if time_stat is not None:
        time_stat.begin("sparse_topk_to_dense_2D_stack")
    dense_v1 = torch.stack(denses)
    if time_stat is not None:
        time_stat.end("sparse_topk_to_dense_2D_stack")
        time_stat.end("sparse_topk_to_dense_2D_fillin")
    #print("dense_v1.shape:", dense_v1.shape)
    #print("dense_v1:", dense_v1)

    #sparse_topk_value_v1 = sparse_topk_value.reshape(orishape[0] * orishape[1])
    #sparse_topk_idx_v1 = sparse_topk_idx.reshape(orishape[0] * orishape[1])

    return dense_v1

def sparse_topk_to_dense_2D0(sparse_topk_value, sparse_topk_idx, dense_ori_dim, device=None, time_stat=None):
    """
    sparse_topk_value: (batchsize, K) topk_value from dense_to_topk_in_sparse
    sparse_topk_idx: (batchsize, K) topk_idx from dense_to_topk_in_sparse
    dense_ori_dim: 1 (the original dim of the original dense matrix)

    input must be normalized to 0~1; otherwise, cannot use sum_of_topk to obtain the remainder
    """

    time_stat.begin("sparse_topk_to_dense_2D_open_dense")
    dim_of_non_topk = 0
    dim_of_topk = -1
    K = sparse_topk_value.shape[dim_of_topk]

    # Step1: init as all 1's
    # dense_v1: (batch_size, N)  N is the original dim of the original dense matrix
    dense_v1 = Variable(torch.ones(sparse_topk_value.shape[dim_of_non_topk], \
        dense_ori_dim).float(), requires_grad=True)
    time_stat.end("sparse_topk_to_dense_2D_open_dense")
    #print("dense_v1:", dense_v1, "device:", device)
    time_stat.begin("sparse_topk_to_dense_2D_dense_tocuda")
    if device == torch.device("cuda"):
        dense_v1 = dense_v1.cuda()
    time_stat.end("sparse_topk_to_dense_2D_dense_tocuda")
    #print("dense_v1:", dense_v1, "device:", device)

    # Step2: init as all (1-sum(prob_of_topk))/ (N-topk)
    # sum_of_topk: (batchsize)
    # remainder: (batchsize)
    time_stat.begin("sparse_topk_to_dense_2D_remainder")
    sum_of_topk = sparse_topk_value.sum(dim=dim_of_topk)
    remainder = (1-sum_of_topk) * 1.0 / (dense_ori_dim - K)
    remainder = Variable(torch.tensor(remainder).float())
    time_stat.end("sparse_topk_to_dense_2D_remainder")
    time_stat.begin("sparse_topk_to_dense_2D_remainder_tocuda")
    remainder.to(device)
    time_stat.end("sparse_topk_to_dense_2D_remainder_tocuda")
    #print("remainder:", remainder)

    # Step3: fill in topk slots
    time_stat.begin("sparse_topk_to_dense_2D_fillin")
    for i in range(dense_v1.shape[0]):
        #print("dense_v1[i]:", dense_v1[i])
        dense_v1[i] = dense_v1[i] * remainder[i]
        #print("dense_v1[i]:", dense_v1[i])
        #for dv, di in zip(sparse_topk_value[i], sparse_topk_idx[i]):
        for idx in range(K):
            dense_v1[i][sparse_topk_idx[i][idx]] = sparse_topk_value[i][idx]
            #print("dense_v1[i][di]:", i, di, dense_v1[i][di])
    time_stat.end("sparse_topk_to_dense_2D_fillin")

    return dense_v1

def sparse_topk_to_dense_3D(sparse_topk_value, sparse_topk_idx, dense_ori_dim, with_softmax=True, device=None, time_stat=None, sum_of_all_values=None, remainder_is_0=False):
    """
    sparse_topk_value: (batchsize, seqlen, K) topk_value from dense_to_topk_in_sparse
    sparse_topk_idx: (batchsize, seqlen, K) topk_idx from dense_to_topk_in_sparse
    dense_ori_dim: 1 (the original dim of the original dense matrix)

    input must be normalized to 0~1; otherwise, cannot use sum_of_topk to obtain the remainder
    """

    sshape = sparse_topk_value.shape
    sparse_topk_value_v1 = sparse_topk_value.reshape(sshape[0] * sshape[1], sshape[2])
    sparse_topk_idx_v1 = sparse_topk_idx.reshape(sshape[0] * sshape[1], sshape[2])
    if sum_of_all_values is not None:
        sum_of_all_values = sum_of_all_values.reshape(sshape[0] * sshape[1])
    
    dense_v1 = sparse_topk_to_dense_2D(sparse_topk_value_v1, sparse_topk_idx_v1, \
        dense_ori_dim, with_softmax=with_softmax, device=device, time_stat=time_stat, \
        sum_of_all_values=sum_of_all_values, remainder_is_0=remainder_is_0)

    dense_v2 = dense_v1.reshape(sshape[0], sshape[1], dense_ori_dim)

    return dense_v2

def union_set_of_list_of_ids(src, device='cpu'):
    """
       for a given list "src", get the union set of each item of the list
       result: Union(src[0], src[1], ..., src[N])
    """
    data0 = torch.stack(src)
    data1 = data0.cpu().tolist()
    data2 = set(data1)
    data3 = torch.tensor(data2, device=device)
    return data3
    
def get_binary_mask_from_sparse_index_3D(indexs, dense_ori_dim, \
        value_of_index=torch.tensor(1.0), time_stat=None, device=None):
    """
    generate a binary tensor, 
        where the values on the given index is value_of_index (defalut=1)
        where the values on other index is 0.0

    indexs: (e.g. batchsize, seqlen, K)  (K comes from top-K)
        but the indexs only occurs on the last dim (dim_of_real_index = -1)

    dense_ori_dim: (e.g. vocab_size) the target dim of output on dim_of_real_index axis
        if (batchsize, seqlen, K) -> (batchsize, seqlen, vocab_size), the dense_ori_dim is vocab_size

    output:
        binary tensor (masked) (e.g. batchsize, seqlen, vocab_size)
    """

    dim_of_real_index = -1
    target_shape = torch.Size(list(indexs.shape[:dim_of_real_index]) + [dense_ori_dim])

    if time_stat is not None:
        time_stat.begin("get_binary_mask_from_sparse_index_3D")
    
    # 3D (batchsize, seqlen, K) -> 2D (batchsize * seqlen, K)
    indexs_shape = indexs.shape
    probs_dims_of_non_real_index = np.prod(list(indexs_shape[:dim_of_real_index]))
    print("dim_of_real_index:", dim_of_real_index, "dense_ori_dim:", dense_ori_dim, "target_shape:", target_shape, "indexs_shape:", indexs_shape, "probs_dims_of_non_real_index:", probs_dims_of_non_real_index)
    indexs2D = indexs.reshape(probs_dims_of_non_real_index, indexs_shape[dim_of_real_index])
    print("indexs2D.shape:", indexs2D.shape)

    K = indexs_shape[dim_of_real_index]
    values = value_of_index.repeat(K)

    mask_list = []
    for this_index in indexs2D:
        if time_stat is not None:
            time_stat.begin("get_binary_mask_from_sparse_index_3D_sparse")
        neg_one = torch.tensor(-1, device=device)
        if neg_one in this_index: # for _get_student_topk_and_mask_others_K_vary_with_sample
            mask = (this_index!=neg_one)
            this_index2 = torch.masked_select(this_index, mask)
            #print("in get_binary_mask_from_sparse_index_3D. mask:", mask, "this_index2:", this_index2)
            new_real_K = this_index2.shape[0]
            this_index2 = this_index2.unsqueeze(0)
            this_mask = torch.sparse.FloatTensor(this_index2, values[:new_real_K], torch.Size([dense_ori_dim]))
        else:
            this_index = this_index.unsqueeze(0)
            this_mask = torch.sparse.FloatTensor(this_index, values, torch.Size([dense_ori_dim]))

        if time_stat is not None:
            time_stat.end("get_binary_mask_from_sparse_index_3D_sparse")
            time_stat.begin("get_binary_mask_from_sparse_index_3D_to_dense")
        this_mask_v2 = this_mask.to_dense()
        # Now: masked (non-topK) value is 0, non masked (topK) value is 1, should reverse

        this_mask_v3 = 1.0 - this_mask_v2
        # Now: masked (non-topK) value is 1, non masked (topK) value is 0, good!

        if time_stat is not None:
            time_stat.end("get_binary_mask_from_sparse_index_3D_to_dense")

        mask_list.append(this_mask_v3)
    mask_v1 = torch.stack(mask_list)
    mask_v2 = mask_v1.reshape(target_shape)

    if time_stat is not None:
        time_stat.end("get_binary_mask_from_sparse_index_3D")

    return mask_v2

def _binary_search(src, lower, cur, upper, threshold):
    if src[0:upper].sum() <= threshold:
        return upper - 1

    #print("lower:", lower, "cur:", cur, "upper:", upper, "threshold:", threshold)
    cur_value = src[0:cur].sum()
    #print("cur_value:", cur_value)
    if torch.abs(cur_value - threshold) < 0.001:
        #print("found abs(cur_value - threshold):", torch.abs(cur_value - threshold))
        return cur
    if (cur - lower) < 2 and (upper - cur) < 2:
        #print("found by index <5:", cur, lower, upper)
        return cur
    if cur_value > threshold:
        if lower >= (lower+cur)//2:
            return lower
        return _binary_search(src, lower, (lower+cur)//2, cur, threshold)
    elif cur_value < threshold:
        if upper <= (lower+cur)//2:
            return upper
        return _binary_search(src, cur, (upper+cur)//2, upper, threshold)
    else:
        return cur

def _get_index_given_threshold_on_cdf_2D(topk_value_candidates, threshold, adaptive_topk_min_k, device):
    """
      topk_value_candidates is the sorted tensor (the return topk_value of torch.topk() function)
      e.g.: topk_value_candidates, topk_index = torch.topk(dense, K=100, dim=dim_of_topk)
          topk_value_candidates: (seqlen, K=100)
      threshold: float 

      Output:
          topk_real_K: (seqlen) each element shows the real "key"
    """
    filter_cnt = 0
    topk_real_K_list = []
    for tvc in topk_value_candidates:
        ori_K = tvc.shape[0]
        try:
            index0 = _binary_search(tvc, 0, ori_K//2, ori_K, threshold)
            if adaptive_topk_min_k > 0 and index0 < adaptive_topk_min_k:
                index = adaptive_topk_min_k
            else:
                index = index0
            filter_cnt += ori_K - index - 1
            print("_binary_search index:", index, "ori index:", index0, "filter_cnt:", filter_cnt, "coverage_when_cut:", tvc[0:index].sum().item())
        except:
            index = ori_K
            print("_binary_search failed. use ori_K:", index)
        topk_real_K_list.append(index)
    topk_real_K = torch.tensor(topk_real_K_list, device=device)
    return topk_real_K, filter_cnt

def _get_index_given_threshold_on_cdf_3D(topk_value_candidates, threshold, adaptive_topk_min_k, device):
    """
      topk_value_candidates is the sorted tensor (the return topk_value of torch.topk() function)
      e.g.: topk_value_candidates, topk_index = torch.topk(dense, K=100, dim=dim_of_topk)
          topk_value_candidates: (batchsize, seqlen, K=100)
      threshold: float 

      Output:
          topk_real_K: (batchsize, seqlen) each element shows the real "key"
    """
    
    topk_real_K_list = []
    filter_cnt_total = 0
    for tvc in topk_value_candidates:
        topk_real_K, filter_cnt = _get_index_given_threshold_on_cdf_2D(tvc, threshold, adaptive_topk_min_k, device)
        topk_real_K_list.append(topk_real_K)
        filter_cnt_total += filter_cnt
        print("filter_cnt:", filter_cnt, "total:", tvc.nelement())
    topk_real_K = torch.stack(topk_real_K_list)
    print("topk_real_K:", topk_real_K, "filter_cnt_total:", filter_cnt_total, "total_ori_topk:", topk_value_candidates.nelement())
    return topk_real_K

def mask_topk_idx_with_topk_real_K(topk_idx, topk_real_K, device):
    """
     mask topk_idx 
     topk_real_K (batch, seqlen), the value shows the real K
     topk_idx (batch, seqlen, ori_K), the value shows the index (for ori K's version)
     OUTPUT: topk_idx (batch, seqlen, ori_K), the value shows the index while -1 as the value shows do not use this slot
    """

    dim_batch = topk_idx.shape[0]
    dim_seq = topk_idx.shape[1]
    dim_k = topk_idx.shape[2]

    print("before mask_topk_idx_with_topk_real_K topk_idx:", topk_idx)
    neg_one = torch.tensor(-1, device=device)
    for i in range(dim_batch):
        for j in range(dim_seq):
            #print("topk_real_K:", topk_real_K)
            real_K = topk_real_K[i][j]
            #print("real_K:", real_K)
            for l in range(int(real_K.item()), dim_k):
                topk_idx[i][j][l] = neg_one
    print("after mask_topk_idx_with_topk_real_K topk_idx:", topk_idx)

def _count_gt_in_student_topk(labels, s_topk_idx, device, mark=""):
    padding_id_in_label = torch.tensor(50256, device=device)
    print("labels:", labels)
    print("s_topk_idx:", s_topk_idx)
    print("labels.shape:", labels.shape)
    print("s_topk_idx.shape:", s_topk_idx.shape)
    labels_1d = labels.reshape(-1, 1).squeeze()
    s_topk_idx_2d = s_topk_idx.reshape(-1, s_topk_idx.shape[-1])
    print("labels_1d.shape:", labels_1d.shape)
    print("s_topk_idx_2d.shape:", s_topk_idx_2d.shape)

    intopk = 0
    total = 0
    for l, s in zip(labels_1d, s_topk_idx_2d):
        if l == padding_id_in_label:
            continue
        if l in s:
            intopk += 1
        total += 1
    print(mark, "gt_in_student_topk_coverage_rate in _get_student_topk_and_mask_others:", intopk*1.0/total, "intopk:", intopk, "total:", total)

def _count_topk_coverage_with_topk_real_K(topk_value, topk_real_K):
    topk_value2 = topk_value.reshape(topk_value.shape[0] * topk_value.shape[1], topk_value.shape[2])
    topk_real_K2 = topk_real_K.reshape(-1)
    coverages = []
    for i, (tv, k) in enumerate(zip(topk_value2, topk_real_K2)):
        coverage = tv[:k].sum()
        print("_count_topk_coverage_with_topk_real_K i:", i, "coverage:", coverage.item())
        coverages.append(coverage)
    coverages2 = torch.stack(coverages).mean()
    print("_count_topk_coverage_with_topk_real_K avg_coverage:", coverages2.item())
    return coverages2.item()

def _mask_1Dvector_in_2Dtensor(src, mask_index, device, mask_value=torch.tensor(1.0)):
    """
       src: (batch * seqlen, vocabsize)
       mask_index: 
       device: gpu or cpu e.g. device=self.device
       mask_value: 0.0 or 1.0  1.0 means non-topk (will be not supervised (will be set s_prob==t_prob so as to not receive supervision)
    """
    #mask_value2 = torch.tensor(mask_value, device=device, requires_grad=False)
    mask_vector = torch.ones(src.shape[-1], device=device, requires_grad=False) * mask_value
    for i, v in enumerate(mask_index):
        if v == 0:
            src[i] = mask_vector
    return src

def _random_or_active_mask_1Dvector_in_3Dtensor(src, topk_index, random_rate, device, mask_value=torch.tensor(1.0), labels_for_select=None, select_threshold=90, train_batchid=-1, queried_sample_token_dict={}, step=-1, t_prob=None, max_uniq_query_num=-1):
    """
       src: the mask matrix, a binary matrix (batch, seqlen, vocabsize)
       topk_index: topk_index from student topk (batch, seqlen, K)
       random_rate: a float (0~1) [0.1 means only 10% steps will be supervised]
       device: gpu or cpu e.g. device=self.device
       mask_value: 0.0 or 1.0  1.0 means non-topk (will be not supervised (will be set s_prob==t_prob so as to not receive supervision))
    labels_for_select: ground-truth (batch, seqlen) ==None means use random pick
    select_threshold: if ground-truth is not in top[select_threshold], should receive supervision 
    t_prob: teacher's probability without noise (batch, seqlen, vocabsize)
    max_uniq_query_num: max query num. if reach the number, use only the existed queries

       OUTPUT:
       res: new src (new mask matrix, a binary matrix)  (batch, seqlen, vocabsize)
       new_topk_index: (batch, seqlen, K) the random masked elements' topk index is -1
    """
    def __print_teacher_result_for_new_query(id_for_print, topk_index, t_prob):
        """
        topk_index: (K)
        t_prob: (vocabsize)
        """
        prob_of_topk = t_prob[topk_index]
        prob_of_topk_str = " ".join(list(map(str, prob_of_topk.tolist())))
        print("QUERY_FOR_ANALYSIS " + id_for_print + " " + prob_of_topk_str)

    src2 = src.reshape(-1, src.shape[-1])
    labels_for_select2 = labels_for_select.reshape(-1)
    topk_index2 = topk_index.reshape(-1, topk_index.shape[-1])
    t_prob2 = t_prob.reshape(-1, t_prob.shape[-1]) # t_prob2: (batch*seqlen, vocabsize)
    if select_threshold != 0 and labels_for_select is not None:
        mask_index = []
        cnt1, cnt2 = 0, 0
        #print("labels_for_select2.shape:", labels_for_select2.shape)
        #print("topk_index2.shape:", topk_index2.shape)
        # works only when batchsize == 1
        for i, (l, tk) in enumerate(zip(labels_for_select2, topk_index2)):
            uniq_query_num = len(queried_sample_token_dict.keys())
            key = str(step) + "_" + str(train_batchid) + "_" + str(i) 

            if l in tk[:select_threshold]:
                mask_index.append(0)
                cnt1 += 1
            elif max_uniq_query_num != -1 and max_uniq_query_num <= uniq_query_num:
                # use max_uniq_query_num and excceed the max number
                if key not in queried_sample_token_dict:
                    mask_index.append(0)
                    cnt1 += 1
                else: # can only use the existed queries
                    mask_index.append(1)
                    cnt2 += 1
            else:
                if key not in queried_sample_token_dict:
                    print("key in _random_or_active_mask_1Dvector_in_3Dtensor:", key)
                    queried_sample_token_dict[key] = 1
                    __print_teacher_result_for_new_query(key, tk, t_prob2[i])
                mask_index.append(1)
                cnt2 += 1
        print("labels_for_select2:", labels_for_select2)
        print("cnt_for_mask=0(not_query):", cnt1, "cnt_for_mask=1(query):", cnt2, "uniq_query_num:", uniq_query_num)
    else:
        random_len = src2.shape[0]
        mask_index = [int(random.random()/random_rate<=1) for i in range(random_len)]
    print("mask_index in _random_or_active_mask_1Dvector_in_3Dtensor:", mask_index)
    print("src2 in _random_or_active_mask_1Dvector_in_3Dtensor:", src2)
    print("topk_index:", topk_index)
    #mask_index = torch.tensor([int(random.random()/random_rate<=1) for i in range(random_len)], device=device)
    res0 = _mask_1Dvector_in_2Dtensor(src2, mask_index, device, mask_value)
    res = res0.reshape(src.shape)

    #new_topk_index0 = _mask_1Dvector_in_2Dtensor(topk_index2, mask_index, device, torch.tensor(-1))
    #new_topk_index = new_topk_index0.reshape(topk_index.shape)
    new_topk_index = topk_index

    print("src.shape:", src.shape, "topk_index.shape:", topk_index.shape)
    print("res0.shape:", res0.shape, "res.shape:", res.shape, "new_topk_index.shape:", new_topk_index.shape)

    print("res in _random_or_active_mask_1Dvector_in_3Dtensor:", res)
    print("new_topk_index:", new_topk_index)
    return res, new_topk_index
