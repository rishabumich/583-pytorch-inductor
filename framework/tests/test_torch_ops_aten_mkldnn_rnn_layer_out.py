import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MkldnnRnnLayer_OutModule(torch.nn.Module):
    def forward(self, input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, out0, out1, out2, out3):
        return torch.ops.aten.mkldnn_rnn_layer.out(input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, out0, out1, out2, out3)

mod = Torch_Ops_Aten_MkldnnRnnLayer_OutModule()

input = torch.randn(3)
weight0 = torch.randn(3)
weight1 = torch.randn(3)
weight2 = torch.randn(3)
weight3 = torch.randn(3)
hx_ = torch.randn(3)
cx_ = torch.randn(3)
reverse = True
batch_sizes = 3
mode = 3
hidden_size = 3
num_layers = 3
has_biases = True
bidirectional = True
batch_first = True
train = True
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)
out3 = torch.randn(3)

args = (input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, out0, out1, out2, out3,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
