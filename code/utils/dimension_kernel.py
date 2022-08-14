import torch
import torch.nn as nn
import numpy as np


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, include_input=True):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)

    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding
class Trigonometric_kernel:
    def __init__(self, L=10, include_input=True):
        self.L = L

        self.embed_fn, self.out_ch = get_embedder(L, include_input=include_input)

    '''
    INPUT
     x: input vectors (N,C) 

     OUTPUT

     pos_kernel: (N, calc_dim(C) )
    '''

    def __call__(self, x):
        return self.embed_fn(x)

    def calc_dim(self, dims=0):
        return self.out_ch


# Fourier encoding
class FourierKernel(nn.Module):
    def __init__(self, in_dim=3, L=10, sigma=10.0, include_input=False):
        super(FourierKernel, self).__init__()
        self.in_dim = in_dim
        self.L = L
        self.sigma = sigma
        self.include_input = include_input
        gaussian_kernel = torch.randn([L, in_dim]) * sigma
        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def forward(self, x):
        embed_pos = []
        if self.include_input:
            embed_pos.append(x)
        x = 2 * np.pi * x
        x = torch.matmul(x, self.gaussian_kernel.T)
        embed_pos.append(torch.sin(x))
        embed_pos.append(torch.cos(x))
        return torch.cat(embed_pos, -1)

    def calc_dim(self, dims=0):
        dim = self.L * 2
        if self.include_input:
            dim += self.in_dim
        return dim


def make_encoding_kernel(in_dim=3, L=10, L1=4, sigma=10., include_input=True, method="POS"):
    if method == "FOURIER":
        return FourierKernel(in_dim=in_dim, L=L, sigma=sigma, include_input=include_input)
    else:
        return Trigonometric_kernel(L=L, include_input=include_input)


def make_dir_encoding_kernel(in_dim=3, L=10, sigma=10., include_input=True, method="POS"):
    if method == "FOURIER":
        return FourierKernel(in_dim=in_dim, L=L, sigma=sigma, include_input=include_input)
    else:
        return Trigonometric_kernel(L=L, include_input=include_input)


if __name__ == "__main__":
    pos = torch.ones([2, 3]).cuda()

    kernel = make_encoding_kernel(in_dim=3, L=20, L1=4, sigma=10., include_input=True, method="MLP").cuda()

    print(kernel.calc_dim(0))
