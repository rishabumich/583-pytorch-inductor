import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedScatterBackwardModule(torch.nn.Module):
    def forward(self, grad_output, mask, sizes):
        return torch.ops.aten.masked_scatter_backward(grad_output, mask, sizes)

mod = Torch_Ops_Aten_MaskedScatterBackwardModule()

grad_output = torch.randn(3)
mask = torch.randn(3)
sizes = torch.tensor(0)  # Fallback for unknown type SymInt[]

args = (grad_output, mask, sizes,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
