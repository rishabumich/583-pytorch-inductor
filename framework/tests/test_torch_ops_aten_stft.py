import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_StftModule(torch.nn.Module):
    def forward(self, x, n_fft, hop_length, win_length, window, normalized, onesided, return_complex):
        return torch.ops.aten.stft(x, n_fft, hop_length, win_length, window, normalized, onesided, return_complex)

mod = Torch_Ops_Aten_StftModule()

x = torch.randn(3)
n_fft = 3
hop_length = 3
win_length = 3
window = torch.randn(3)
normalized = True
onesided = True
return_complex = True

args = (x, n_fft, hop_length, win_length, window, normalized, onesided, return_complex,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
