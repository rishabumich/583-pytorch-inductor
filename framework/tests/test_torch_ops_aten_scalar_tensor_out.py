import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ScalarTensor_OutModule(torch.nn.Module):
    def forward(self, s, out):
        return torch.ops.aten.scalar_tensor.out(s, out=out)

mod = Torch_Ops_Aten_ScalarTensor_OutModule()

s = None  # Fallback for unknown type |Scalar
out = torch.empty(3)

args = (s, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
