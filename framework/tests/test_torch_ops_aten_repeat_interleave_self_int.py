import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RepeatInterleave_SelfIntModule(torch.nn.Module):
    def forward(self, x, repeats, dim, output_size):
        return torch.ops.aten.repeat_interleave.self_int(x, repeats, dim, output_size)

mod = Torch_Ops_Aten_RepeatInterleave_SelfIntModule()

x = torch.randn(3)
repeats = None  # Fallback for unknown type SymInt
dim = 3
output_size = torch.sym_int(3)

args = (x, repeats, dim, output_size,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
