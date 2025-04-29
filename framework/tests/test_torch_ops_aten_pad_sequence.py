import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PadSequenceModule(torch.nn.Module):
    def forward(self, sequences, batch_first, padding_value, padding_side):
        return torch.ops.aten.pad_sequence(sequences, batch_first, padding_value, padding_side)

mod = Torch_Ops_Aten_PadSequenceModule()

sequences = torch.randn(3)
batch_first = True
padding_value = 1.0
padding_side = torch.tensor(0)  # Fallback for unknown type str

args = (sequences, batch_first, padding_value, padding_side,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
