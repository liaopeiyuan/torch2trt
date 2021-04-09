from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.reshape')
def convert_reshape(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    #print(input_a)
    #print(ctx.method_return)
    input_a_trt = trt_(ctx.network, input_a)
    output = ctx.method_return
    dims = trt.Dims(input_b)
    #print(input_b)
    layer = ctx.network.add_shuffle(input_a_trt)
    layer.reshape_dims = dims
    output._trt = layer.get_output(0)
