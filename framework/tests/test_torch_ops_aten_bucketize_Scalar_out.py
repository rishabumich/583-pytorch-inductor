import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Bucketize_ScalarOutModule(torch.nn.Module):
    def forward(self, x, boundaries, out_int32, right, out):
        return torch.ops.aten.bucketize.Scalar_out(x, boundaries, out_int32, right, out=out)

mod = Torch_Ops_Aten_Bucketize_ScalarOutModule()

x = torch.tensor(0)  # Fallback for unknown type |Scalar
boundaries = torch.randn(3)
out_int32 = True
right = True
out = torch.empty(3)

args = (x, boundaries, out_int32, right, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
