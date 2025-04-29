import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds
import sys

# Redirect all stdout to a file
original_std_out = sys.stdout
#sys.stdout = open("output_torch.ops.aten.acosh.out.txt", "w")

# Define a module with the operator
class Torch_Ops_Aten_Acosh_OutModule(torch.nn.Module):
    def forward(self,in1,out):
        return torch.ops.aten.acosh.out(in1,out=out)

# Instantiate the model
mod = Torch_Ops_Aten_Acosh_OutModule()

# Define inputs
in1 = torch.randn(3,3)
out = torch.randn(3,3)

args = (in1,out,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)

sys.stdout = original_std_out
