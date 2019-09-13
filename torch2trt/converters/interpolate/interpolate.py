import tensorrt as trt
import torch.nn.functional as F
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .interpolate_plugin import *


@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    try:
        mode = ctx.method_kwargs['mode']
    except KeyError:
        mode = 'nearest'

    try:
        align_corners = ctx.method_kwargs['align_corners']
    except KeyError:
        align_corners = False

    # currently only works for NCHW
    size = list(output.shape[2:])

    interpolate_plugin.add_to_network(ctx.network, [input], [output], size=size, mode=mode, align_corners=align_corners)


class Interpolate(torch.nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_nearest():
    return Interpolate((224, 224), 'nearest', None)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_bilinear():
    return Interpolate((224, 224), 'bilinear', False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_bicubic():
    return Interpolate((224, 224), 'bicubic', False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_area():
    return Interpolate((56, 56), 'area', None)
