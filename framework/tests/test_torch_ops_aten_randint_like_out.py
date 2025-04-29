import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RandintLike_OutModule(torch.nn.Module):
    def forward(self, x, high, memory_format, out):
        return torch.ops.aten.randint_like.out(x, high, memory_format, out=out)

mod = Torch_Ops_Aten_RandintLike_OutModule()

x = torch.randn(3)
high = torch.tensor(0)  # Fallback for unknown type SymInt
memory_format = torch.tensor(0)  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (x, high, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
