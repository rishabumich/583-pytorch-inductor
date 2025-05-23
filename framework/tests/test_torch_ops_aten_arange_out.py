import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Arange_OutModule(torch.nn.Module):
    def forward(self, end, out):
        return torch.ops.aten.arange.out(end, out=out)

mod = Torch_Ops_Aten_Arange_OutModule()

end = None  # Fallback for unknown type |Scalar
out = torch.empty(3)

args = (end, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
