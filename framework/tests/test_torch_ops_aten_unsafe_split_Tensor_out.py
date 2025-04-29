import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UnsafeSplit_TensorOutModule(torch.nn.Module):
    def forward(self, x, split_size, dim, out):
        return torch.ops.aten.unsafe_split.Tensor_out(x, split_size, dim, out=out)

mod = Torch_Ops_Aten_UnsafeSplit_TensorOutModule()

x = torch.randn(3)
split_size = torch.tensor(0)  # Fallback for unknown type SymInt
dim = 3
out = torch.empty(3)

args = (x, split_size, dim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
