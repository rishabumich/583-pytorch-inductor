import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randn_GeneratorWithNamesOutModule(torch.nn.Module):
    def forward(self, size, generator, names, out):
        return torch.ops.aten.randn.generator_with_names_out(size, generator, names, out=out)

mod = Torch_Ops_Aten_Randn_GeneratorWithNamesOutModule()

size = torch.sym_int(3)
generator = None  # Fallback for unknown type Generator?
names = None  # Fallback for unknown type str[]?
out = torch.empty(3)

args = (size, generator, names, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
