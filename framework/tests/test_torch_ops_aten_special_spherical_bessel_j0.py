import torch
from torch._decomp import decomposition_table

op = torch.ops.aten.special_spherical_bessel_j0

x = torch.randn(3)

print("Scalar-returning op:")
if op in decomposition_table:
    print(f"{op} is decomposed.")
else:
    print(f"{op} is NOT decomposed.")

result = op(x)
print("Result:", result)
