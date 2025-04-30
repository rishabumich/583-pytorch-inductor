import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ReplicationPad1DModule(torch.nn.Module):
    def forward(self, x, padding):
        return torch.ops.aten.replication_pad1d(x, padding)

mod = Torch_Ops_Aten_ReplicationPad1DModule()

x = torch.randn(3)
padding = torch.sym_int(3)

args = (x, padding,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
