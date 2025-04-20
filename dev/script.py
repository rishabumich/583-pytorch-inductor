import ast
import copy
import sys

# ============================================================
# STEP 1: Extract function signatures from decompositions.py
# ============================================================
#
# We parse decompositions.py to get each function's parameters.
# The idea is to build a dictionary:
#   signatures = {
#       "function_name": [ (param1, default1), (param2, default2), ... ]
#   }
#
# For each function, if a parameter is required (i.e. no default),
# we store its default as None. For optional parameters, we store the AST node
# representing its default value.
#

def extract_signatures(decompositions_filename):
    with open(decompositions_filename, "r") as f:
        code = f.read()
    tree = ast.parse(code, mode="exec")
    signatures = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Get the list of argument names (we assume no *args/**kwargs for simplicity)
            arg_names = [arg.arg for arg in node.args.args]
            defaults = node.args.defaults
            num_args = len(arg_names)
            num_defaults = len(defaults)
            sig = []
            # The defaults correspond to the last N parameters (if any)
            for i, arg in enumerate(arg_names):
                if i < num_args - num_defaults:
                    # Required parameter; no default exists.
                    sig.append( (arg, None) )
                else:
                    # The default value for this parameter from the function definition.
                    default_index = i - (num_args - num_defaults)
                    default_value_node = defaults[default_index]
                    # Make a deep copy so that later modifications to the AST
                    # in the function calls file do not affect these nodes.
                    sig.append( (arg, copy.deepcopy(default_value_node)) )
            signatures[func_name] = sig
    return signatures

# ============================================================
# STEP 2: Create an AST transformer to fix function calls.
# ============================================================
#
# This NodeTransformer will visit every call in all_function_calls.py. If the 
# target function (by name) is found in our signatures dictionary, then for each 
# provided argument (both positional and keyword), if its value is the literal None,
# then it replaces that argument with:
#
#    - the default value from the signature (if one is defined) or
#    - a dummy placeholder (here we use ast.Constant(value=0)) if the parameter is required.
#
# In a production scenario you might implement a more sophisticated choice for placeholder
# (e.g. based on parameter names or type annotations).
#

class CallFixer(ast.NodeTransformer):
    def __init__(self, signatures):
        self.signatures = signatures
        super().__init__()

    def visit_Call(self, node):
        self.generic_visit(node)
        # We only handle simple calls (where node.func is a Name)
        if isinstance(node.func, ast.Name) and node.func.id in self.signatures:
            func_name = node.func.id
            sig = self.signatures[func_name]
            # Process positional arguments.
            # Note: We assume the call uses only positional arguments for the first N params.
            new_args = []
            for i, arg in enumerate(node.args):
                # Only fix if we have a matching parameter (extra arguments are left as-is)
                if i < len(sig):
                    param_name, default_value = sig[i]
                    if isinstance(arg, ast.Constant) and arg.value is None:
                        # If there is a default value in the signature, use it.
                        if default_value is not None:
                            # We use a copy so that subsequent modifications do not conflict.
                            new_args.append(copy.deepcopy(default_value))
                        else:
                            # For required parameters that were incorrectly set to None,
                            # we use a dummy placeholder. (Here we choose 0.)
                            new_args.append(ast.Constant(value=0))
                    else:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            node.args = new_args

            # Process keyword arguments.
            for kw in node.keywords:
                # Find the parameter in the signature corresponding to this keyword.
                match = None
                for param_name, default_value in sig:
                    if param_name == kw.arg:
                        match = (param_name, default_value)
                        break
                if match is not None:
                    _, default_value = match
                    if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                        if default_value is not None:
                            kw.value = copy.deepcopy(default_value)
                        else:
                            kw.value = ast.Constant(value=0)
        return node

# ============================================================
# STEP 3: Putting It All Together
# ============================================================
#
# This function reads the original all_function_calls file, applies our transformation,
# and writes the modified version to a new file so that you can compile and investigate.
#

def fix_function_calls(decompositions_filename, calls_filename, output_filename):
    signatures = extract_signatures(decompositions_filename)
    with open(calls_filename, "r") as f:
        calls_code = f.read()
    calls_tree = ast.parse(calls_code, mode="exec")

    # Transform the AST using our CallFixer.
    fixer = CallFixer(signatures)
    fixed_tree = fixer.visit(calls_tree)
    ast.fix_missing_locations(fixed_tree)

    # Generate the modified code (requires Python 3.9+ for ast.unparse).
    fixed_code = ast.unparse(fixed_tree)
    with open(output_filename, "w") as f:
        f.write(fixed_code)
    print(f"Modified function calls written to: {output_filename}")

# ============================================================
# Usage Example
# ============================================================
#
# If you want to run this script from the command line:
#
#   python fix_calls.py decompositions.py all_function_calls.py fixed_calls.py
#
# it will read decompositions.py and all_function_calls.py, fix the calls as described,
# and write the output to fixed_calls.py.
#

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fix_calls.py <decompositions.py> <all_function_calls.py> <output.py>")
        sys.exit(1)
    decompositions_file = sys.argv[1]
    calls_file = sys.argv[2]
    output_file = sys.argv[3]
    fix_function_calls(decompositions_file, calls_file, output_file)
