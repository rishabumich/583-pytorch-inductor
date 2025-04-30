import os

def generate_test(fn_name, arg_names, arg_types, return_type, onlySelf):
    class_name = fn_name.title().replace("_", "").replace(".", "_") + "Module"

    adjusted_arg_names = []
    tensor_placeholders = []
    call_args_list = []

    for name, typ in zip(arg_names, arg_types):
        if name == "self":
            adjusted_name = "x"
        else:
            adjusted_name = name
        adjusted_arg_names.append(adjusted_name)

        # Improved Tensor type detection
        if "Tensor" in typ or "Tensor[" in typ or "Tensor?" in typ:
            if name == "out":
                tensor_placeholders.append(f"{adjusted_name} = torch.empty(3)")
                call_args_list.append(f"{adjusted_name}={adjusted_name}")
            else:
                tensor_placeholders.append(f"{adjusted_name} = torch.randn(3)")
                call_args_list.append(f"{adjusted_name}")
        elif typ == "float" or "float[" in typ or "float?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = 1.0")
            call_args_list.append(f"{adjusted_name}")
        elif typ == "int" or "int[" in typ or "int?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = 3")
            call_args_list.append(f"{adjusted_name}")
        elif typ == "bool" or "bool[" in typ or "bool?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = True")
            call_args_list.append(f"{adjusted_name}")
        elif typ == "complex" or "complex[" in typ or "complex?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = complex(1.0, 2.0)")
            call_args_list.append(f"{adjusted_name}")
        elif typ == "Scalar" or "Scalar[" in typ or "Scalar?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = 1")
            call_args_list.append(f"{adjusted_name}")
        elif typ == "SymInt?" or "SymInt[" in typ or "SymInt?" in typ:
            tensor_placeholders.append(f"{adjusted_name} = torch.sym_int(3)")
            call_args_list.append(f"{adjusted_name}")
        else:
            # Safe fallback for unknown types such as ScalarType?
            tensor_placeholders.append(f"{adjusted_name} = None  # Fallback for unknown type {typ}")
            call_args_list.append(f"{adjusted_name}")

    call_args = ", ".join(call_args_list)

    # Handle input definitions and args
    if onlySelf:
        inputs_code = "x = torch.randn(3, 3)"
        args_tuple = "(x,)"
    else:
        inputs_code = "\n".join(tensor_placeholders)
        args_tuple = "(" + ", ".join(adjusted_arg_names) + ",)"

    # Tensor-returning ops
    if return_type.startswith("Tensor"):
        return f'''import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class {class_name}(torch.nn.Module):
    def forward(self, {", ".join(adjusted_arg_names) if adjusted_arg_names else 'x'}):
        return {fn_name}({call_args if call_args else 'x'})

mod = {class_name}()

{inputs_code}

args = {args_tuple}

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
'''
    # Scalar-returning ops — Unified output format
    else:
        return f'''import torch
from torch._decomp import decomposition_table

op = {fn_name}

{inputs_code}

print("Scalar-returning op:")
if op in decomposition_table:
    print(f"{{op}} is decomposed.")
else:
    print(f"{{op}} is NOT decomposed.")

result = op({call_args})
print("Result:", result)
'''

def parse_signature_line(line):
    line = line.strip()

    idx = line.find(")")
    if idx == -1:
        raise ValueError(f"Invalid format, missing ')': {line}")

    fn_and_args = line[:idx + 1]
    types_str = line[idx + 1:]

    # Safer split using partition
    fn_name_part, _, args_str = fn_and_args.partition("(")
    fn_name = fn_name_part.strip()

    raw_args = args_str.rstrip(")")
    arg_names = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    arg_types = [typ.strip() for typ in types_str.split(",")]

    onlySelf = (arg_names == ['self'])
    if onlySelf:
        arg_names = []
        arg_types = []

    return fn_name, arg_names, arg_types, arg_types[0] if len(arg_types) == 1 else "Tensor", onlySelf

if __name__ == "__main__":
    with open("signatures.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                fn_name, arg_names, arg_types, return_type, onlySelf = parse_signature_line(line)
                test_code = generate_test(fn_name, arg_names, arg_types, return_type, onlySelf)
                os.makedirs("tests", exist_ok=True)
                filename = f"tests/test_{fn_name.replace('.', '_')}.py"
                with open(filename, "w") as out_file:
                    out_file.write(test_code)
                print(f"Generated: {filename}")
            except Exception as e:
                print(f"Skipping line due to error: {line.strip()}\nReason: {e}")
