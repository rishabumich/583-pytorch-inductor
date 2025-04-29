import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Arange_StartOutModule(torch.nn.Module):
    def forward(self, start, end, step, out):
        return torch.ops.aten.arange.start_out(start, end, step, out=out)

mod = Torch_Ops_Aten_Arange_StartOutModule()

start = torch.tensor(0)  # Fallback for unknown type |Scalar
end = torch.tensor(0)  # Fallback for unknown type Scalar
step = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (start, end, step, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
