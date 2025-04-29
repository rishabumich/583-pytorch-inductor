import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randn_GeneratorOutModule(torch.nn.Module):
    def forward(self, size, generator, out):
        return torch.ops.aten.randn.generator_out(size, generator, out=out)

mod = Torch_Ops_Aten_Randn_GeneratorOutModule()

size = torch.tensor(0)  # Fallback for unknown type |SymInt[]
generator = torch.tensor(0)  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (size, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
