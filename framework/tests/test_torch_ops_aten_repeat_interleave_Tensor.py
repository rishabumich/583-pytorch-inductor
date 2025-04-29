import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RepeatInterleave_TensorModule(torch.nn.Module):
    def forward(self, repeats, output_size):
        return torch.ops.aten.repeat_interleave.Tensor(repeats, output_size)

mod = Torch_Ops_Aten_RepeatInterleave_TensorModule()

repeats = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt?

args = (repeats, output_size,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
