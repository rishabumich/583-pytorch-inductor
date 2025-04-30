import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_QuantizedLstm_DataLegacyModule(torch.nn.Module):
    def forward(self, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic):
        return torch.ops.aten.quantized_lstm.data_legacy(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic)

mod = Torch_Ops_Aten_QuantizedLstm_DataLegacyModule()

data = torch.randn(3)
batch_sizes = torch.randn(3)
hx = torch.randn(3)
params = torch.randn(3)
has_biases = True
num_layers = 3
dropout = 1.0
train = True
bidirectional = True
dtype = None  # Fallback for unknown type ScalarType?
use_dynamic = True

args = (data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
