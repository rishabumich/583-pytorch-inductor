import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MkldnnRnnLayerBackward_OutModule(torch.nn.Module):
    def forward(self, input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace, out0, out1, out2, out3, out4, out5, out6):
        return torch.ops.aten.mkldnn_rnn_layer_backward.out(input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace, out0, out1, out2, out3, out4, out5, out6)

mod = Torch_Ops_Aten_MkldnnRnnLayerBackward_OutModule()

input = torch.randn(3)
weight1 = torch.randn(3)
weight2 = torch.randn(3)
weight3 = torch.randn(3)
weight4 = torch.randn(3)
hx_ = torch.randn(3)
cx_tmp = torch.randn(3)
output = torch.randn(3)
hy_ = torch.randn(3)
cy_ = torch.randn(3)
grad_output = torch.randn(3)
grad_hy = torch.randn(3)
grad_cy = torch.randn(3)
reverse = True
mode = 3
hidden_size = 3
num_layers = 3
has_biases = True
train = True
bidirectional = True
batch_sizes = torch.tensor(0)  # Fallback for unknown type int[]
batch_first = True
workspace = torch.randn(3)
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)
out3 = torch.randn(3)
out4 = torch.randn(3)
out5 = torch.randn(3)
out6 = torch.randn(3)

args = (input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace, out0, out1, out2, out3, out4, out5, out6,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
