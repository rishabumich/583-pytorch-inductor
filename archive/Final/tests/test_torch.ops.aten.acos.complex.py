import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds
import sys

# Redirect all stdout to a file
original_std_out = sys.stdout
#sys.stdout = open("output_torch.ops.aten.acos.complex.txt", "w")

# Define a module with the operator
class Torch_Ops_Aten_Acos_ComplexModule(torch.nn.Module):
    def forward(self,a):
        return torch.ops.aten.acos.complex(a)

# Instantiate the model
mod = Torch_Ops_Aten_Acos_ComplexModule()

# Define inputs
a = torch.complex(torch.rand(1),torch.rand(1))

args = (a,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)

sys.stdout = original_std_out
