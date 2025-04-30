import torch
from torch._decomp import decomposition_table

op = torch.ops.aten.expm1.int

a = None  # Fallback for unknown type |int

print("Scalar-returning op:")
if op in decomposition_table:
    print(f"{op} is decomposed.")
else:
    print(f"{op} is NOT decomposed.")

result = op(a)
print("Result:", result)
