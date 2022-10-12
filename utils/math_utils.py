"""
Utils of tensor
"""

import math,random,sys,os
import texar.torch as tx
import torch
import torch.nn as nn
from torch.autograd import Variable

class Laplacian():
    def __init__(self, mean=0, scale=1, device="cpu", \
        use_cache=0, vocab_size=50000, \
        batch_size=32, max_seq_len=128):
        """
        mean is \mu; scale is b in the formula
        """
    
        self.m = torch.distributions.laplace.Laplace(torch.tensor(float(mean)), torch.tensor(float(scale)))
        self.device = device
        self.use_cache = use_cache
        #self.cache_size = cache_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.step = 0
        #self.update_cache_step_for_level0 = 500
        self.update_cache_step_for_level0 = 500
        self.update_cache_step_for_level1 = 5

    def _generate_cache_if_need(self, time_stat):
        if self.use_cache == 0:
            return

        self.cache_dim0 = self.vocab_size
        self.cache_dim1 = self.max_seq_len
        self.cache_dim2 = self.batch_size

        time_stat.begin("_generate_cache_if_need_level0")
        if self.step % self.update_cache_step_for_level0 == 0:
            self.cache0 = []
            self.cache = []
            for j in range(self.cache_dim1):
                #print("level 0 j:", j, "step:", self.step)
                self.cache0.append(torch.tensor([self.m.sample() for i in range(self.cache_dim0)], requires_grad=False))
        time_stat.end("_generate_cache_if_need_level0")

        time_stat.begin("_generate_cache_if_need_level1")
        if self.step % self.update_cache_step_for_level0 == 0 or \
            self.step % self.update_cache_step_for_level1 == 0:
            self.cache = []
            for j in range(self.cache_dim2 * self.cache_dim1):
                #print("level 1 j:", j, "step:", self.step, "self.cache_dim1:", self.cache_dim1)
                index = torch.randperm(self.cache_dim0)
                #print("len(self.cache0):", len(self.cache0), "index:", index)
                index_of_t = random.randint(0, self.cache_dim1 - 1)
                #print("index_of_t:", index_of_t, "len(self.cache0):", len(self.cache0), "index:", index, "len(self.cache0[index_of_t]):", len(self.cache0[index_of_t]))
                data_after_random = self.cache0[index_of_t][index]
                self.cache.append(data_after_random)
        time_stat.end("_generate_cache_if_need_level1")

        self.step += 1
        
    def sample(self):
        return self.m.sample()

    def sample_with_abs(self):
        return self.m.sample().abs()

    def sample_and_add_noise_on(self, data):
        n = data.nelement()
        noise = torch.tensor([self.m.sample() for i in range(n)], device=self.device, requires_grad=False)
        noise2 = noise.reshape(data.shape)
        data_w_n = data + noise2
        return data_w_n

    def sample_and_add_noise_on_with_cache(self, data, time_stat):
        if self.use_cache == 0:
            return None

        time_stat.begin("_generate_cache_if_need")
        self._generate_cache_if_need(time_stat)
        time_stat.end("_generate_cache_if_need")
        time_stat.begin("copy_noise")
        dim1 = data.nelement() // self.cache_dim0
        print("data.shape:", data.shape, "data.nelement():", data.nelement(), "self.cache_dim0:", self.cache_dim0, "self.cache_dim1:", self.cache_dim1, "self.cache_dim2:", self.cache_dim2, "dim1:", dim1)
        noise = []
        for i in range(dim1):
            index_of_t = random.randint(0, self.cache_dim1 * self.cache_dim2 - 1)
            noise.append(self.cache[index_of_t])
        noise2 = torch.stack(noise)
        noise3 = noise2.reshape(data.shape)
        noise3 = noise3.to(self.device)
        data_w_n = data + noise3
        time_stat.end("copy_noise")

        return data_w_n
