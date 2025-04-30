import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Bernoulli_TensorOutModule(torch.nn.Module):
    def forward(self, x, p, generator, out):
        return torch.ops.aten.bernoulli.Tensor_out(x, p, generator, out=out)

mod = Torch_Ops_Aten_Bernoulli_TensorOutModule()

x = torch.randn(3)
p = torch.randn(3)
generator = None  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (x, p, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
