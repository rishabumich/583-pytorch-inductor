import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IstftModule(torch.nn.Module):
    def forward(self, x, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex):
        return torch.ops.aten.istft(x, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex)

mod = Torch_Ops_Aten_IstftModule()

x = torch.randn(3)
n_fft = 3
hop_length = torch.tensor(0)  # Fallback for unknown type int?
win_length = torch.tensor(0)  # Fallback for unknown type int?
window = torch.randn(3)
center = True
normalized = True
onesided = torch.tensor(0)  # Fallback for unknown type bool?
length = torch.tensor(0)  # Fallback for unknown type int?
return_complex = True

args = (x, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
