import torch
import torch.nn as nn
from torch.autograd import Variable

import tensor_utils as tensor_utils
#import utils.tensor_utils as tensor_utils

a=torch.rand([10,20])
b=torch.rand([3,4])

def test_dense_to_topk_in_sparse(data):
    data1v, data1i, data1d = tensor_utils.dense_to_topk_in_sparse(data, K=2)
    data2=tensor_utils.sparse_topk_to_dense(data1v, data1i, data1d)
    print("ori dense:", data)
    print("sparse value:", data1v)
    print("sparse index:", data1i)
    print("sparse dim:", data1d)
    print("dense topk:", data2)

test_dense_to_topk_in_sparse(b)
#test_dense_to_topk_in_sparse(a)
