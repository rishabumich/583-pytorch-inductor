import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Bernoulli_TensorModule(torch.nn.Module):
    def forward(self, x, p, generator):
        return torch.ops.aten.bernoulli_.Tensor(x, p, generator)

mod = Torch_Ops_Aten_Bernoulli_TensorModule()

x = torch.randn(3)
p = torch.randn(3)
generator = torch.tensor(0)  # Fallback for unknown type Generator?

args = (x, p, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
