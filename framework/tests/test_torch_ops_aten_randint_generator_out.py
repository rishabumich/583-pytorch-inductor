import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randint_GeneratorOutModule(torch.nn.Module):
    def forward(self, high, size, generator, out):
        return torch.ops.aten.randint.generator_out(high, size, generator, out=out)

mod = Torch_Ops_Aten_Randint_GeneratorOutModule()

high = torch.tensor(0)  # Fallback for unknown type |SymInt
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
generator = torch.tensor(0)  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (high, size, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
