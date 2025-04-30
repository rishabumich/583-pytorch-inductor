import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RreluWithNoise_OutModule(torch.nn.Module):
    def forward(self, x, noise, lower, upper, training, generator, out):
        return torch.ops.aten.rrelu_with_noise.out(x, noise, lower, upper, training, generator, out=out)

mod = Torch_Ops_Aten_RreluWithNoise_OutModule()

x = torch.randn(3)
noise = torch.randn(3)
lower = 1
upper = 1
training = True
generator = None  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (x, noise, lower, upper, training, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
