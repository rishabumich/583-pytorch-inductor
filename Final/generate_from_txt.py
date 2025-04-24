import os

def generate_test(fn_name, arg_names, arg_types, onlySelf):
    class_name = fn_name.title().replace("_", "").replace(".","_") + "Module"

    tensor_placeholders = []
    for name, typ in zip(arg_names, arg_types):
        if name == "out_grad":
            tensor_placeholders.append(f"{name} = torch.randn(3, requires_grad=True)")
        elif typ == "Tensor" or typ == "Tensor(a!)":
            tensor_placeholders.append(f"{name} = torch.randn(3, requires_grad=True)")
        elif typ == "float" or typ == "Scalar":
            tensor_placeholders.append(f"{name} = 1.0")
        elif typ == "bool":
            tensor_placeholders.append(f"{name} = True")
        elif typ == "int":
            tensor_placeholders.append(f"{name} = torch.randint(low=0, high=10, size=(1,))")
        elif typ == "complex":
            tensor_placeholders.append(f"{name} = torch.complex(torch.rand(1),torch.rand(1))")
        else:
            tensor_placeholders.append(f"{name} = None  # Unknown type")

    call_args = ", ".join(arg_names)
    if(onlySelf):
        func_call = f"x.{fn_name.split(".")[-1]}()"
        forward_args = "x"
        args_tuple = "(in1,)"
        inputs_code = "in1 = torch.randn(3,3)"
    else:
        func_call = f"{fn_name}({call_args})"
        forward_args = ", ".join(arg_names)
        args_tuple = "(" + ", ".join(arg_names) + ",)"
        inputs_code = "\n".join(tensor_placeholders)

    return f'''import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds
import sys

# Redirect all stdout to a file
original_std_out = sys.stdout
#sys.stdout = open("output_{fn_name}.txt", "w")

# Define a module with the operator
class {class_name}(torch.nn.Module):
    def forward(self,{forward_args}):
        return {func_call}

# Instantiate the model
mod = {class_name}()

# Define inputs
{inputs_code}

args = {args_tuple}

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)

sys.stdout = original_std_out
'''


def parse_signature_line(line):
    fn_and_args, type_str = line.strip().split(")|")
    fn_name, args_str = fn_and_args.split("(")
    arg_names = [arg.strip() for arg in args_str.split(",")]
    arg_types = [typ.strip() for typ in type_str.split(",")]
    #Filter out "self"
    if (arg_names[0] == 'self'):
        arg_names = arg_names[1:]
        arg_types = arg_types[1:]
    #Check if self was the only arg
    onlySelf = False
    if (len(arg_names) == 0):
        onlySelf = True
    return fn_name, arg_names, arg_types, onlySelf

if __name__ == "__main__":
    with open("signatures.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue
            fn_name, arg_names, arg_types, onlySelf = parse_signature_line(line)
            test_code = generate_test(fn_name, arg_names, arg_types, onlySelf)
            os.makedirs("tests", exist_ok=True)
            filename = f"tests/test_{fn_name}.py"
            with open(filename, "w") as out_file:
                out_file.write(test_code)
            print(f"Generated: {filename}")
