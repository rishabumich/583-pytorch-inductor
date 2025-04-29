import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ConstantPadNd_OutModule(torch.nn.Module):
    def forward(self, x, pad, value, out):
        return torch.ops.aten.constant_pad_nd.out(x, pad, value, out=out)

mod = Torch_Ops_Aten_ConstantPadNd_OutModule()

x = torch.randn(3)
pad = torch.tensor(0)  # Fallback for unknown type SymInt[]
value = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, pad, value, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
