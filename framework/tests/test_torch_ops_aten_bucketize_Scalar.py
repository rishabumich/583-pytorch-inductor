import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Bucketize_ScalarModule(torch.nn.Module):
    def forward(self, x, boundaries, out_int32, right):
        return torch.ops.aten.bucketize.Scalar(x, boundaries, out_int32, right)

mod = Torch_Ops_Aten_Bucketize_ScalarModule()

x = None  # Fallback for unknown type |Scalar
boundaries = torch.randn(3)
out_int32 = True
right = True

args = (x, boundaries, out_int32, right,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
