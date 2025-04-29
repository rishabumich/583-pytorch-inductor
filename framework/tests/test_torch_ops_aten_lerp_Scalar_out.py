import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Lerp_ScalarOutModule(torch.nn.Module):
    def forward(self, x, end, weight, out):
        return torch.ops.aten.lerp.Scalar_out(x, end, weight, out=out)

mod = Torch_Ops_Aten_Lerp_ScalarOutModule()

x = torch.randn(3)
end = torch.randn(3)
weight = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, end, weight, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
