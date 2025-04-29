import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds
import sys

# Redirect all stdout to a file
original_std_out = sys.stdout
#sys.stdout = open("output_torch.ops.aten.acosh.txt", "w")

# Define a module with the operator
class Torch_Ops_Aten_AcoshModule(torch.nn.Module):
    def forward(self,x):
        return x.acosh()

# Instantiate the model
mod = Torch_Ops_Aten_AcoshModule()

# Define inputs
in1 = torch.randn(3,3)

args = (in1,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)

sys.stdout = original_std_out
