import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_QuantizedGru_DataModule(torch.nn.Module):
    def forward(self, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional):
        return torch.ops.aten.quantized_gru.data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional)

mod = Torch_Ops_Aten_QuantizedGru_DataModule()

data = torch.randn(3)
batch_sizes = torch.randn(3)
hx = torch.randn(3)
params = torch.tensor(0)  # Fallback for unknown type __torch__.torch.classes.rnn.CellParamsBase[]
has_biases = True
num_layers = 3
dropout = 1.0
train = True
bidirectional = True

args = (data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
