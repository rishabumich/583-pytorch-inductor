import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Slice_TensorModule(torch.nn.Module):
    def forward(self, x, dim, start, end, step):
        return torch.ops.aten.slice.Tensor(x, dim, start, end, step)

mod = Torch_Ops_Aten_Slice_TensorModule()

x = torch.randn(3)
dim = 3
start = torch.sym_int(3)
end = torch.sym_int(3)
step = None  # Fallback for unknown type SymInt

args = (x, dim, start, end, step,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
