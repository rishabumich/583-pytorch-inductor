import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_QuantizedGru_InputModule(torch.nn.Module):
    def forward(self, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
        return torch.ops.aten.quantized_gru.input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

mod = Torch_Ops_Aten_QuantizedGru_InputModule()

input = torch.randn(3)
hx = torch.randn(3)
params = torch.tensor(0)  # Fallback for unknown type __torch__.torch.classes.rnn.CellParamsBase[]
has_biases = True
num_layers = 3
dropout = 1.0
train = True
bidirectional = True
batch_first = True

args = (input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
