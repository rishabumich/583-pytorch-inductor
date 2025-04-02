import torch
import torch.utils.benchmark

s = set()

for i in range(6, 15):
    s.add(2**i)
    for j in range(6, i):
        s.add(2**i + 2**j)

ms = [i for i in sorted(s) if i <= 2**14]
ns = [i for i in sorted(s) if i <= 2**14]
ks = [2**i for i in range(10, 14)]

def make_graph(n_iters, f):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_iters):
            f()
    return g

def rowwise_scale(t, dtype_t):
    min_v, max_v = torch.finfo(dtype_t).min, torch.finfo(dtype_t).max
    scale_t = torch.clamp(t.abs().amax(dim=-1, keepdim=True).float(), min=1e-12) / max_v
    t_fp8 = (t / scale_t).clamp(min=min_v, max=max_v).to(dtype_t)
    return t_fp8, scale_t

for m in ms:
    for n in ns:
        for k in ks:
            a = torch.randn((m, k), device="cuda", dtype=torch.float)
            b_t = torch.randn((n, k), device="cuda", dtype=torch.float)
            a_fp8, scale_a = rowwise_scale(a, torch.float8_e4m3fn)
            b_t_fp8, scale_b_t = rowwise_scale(b_t, torch.float8_e4m3fn)
            func = lambda: torch._scaled_mm(
                a_fp8,
                b_t_fp8.t(),
                scale_a=scale_a,
                scale_b=scale_b_t.t(),
                bias=None,
                use_fast_accum=True,
                out_dtype=torch.bfloat16
            )
            print(f"{m=},{n=},{k=}")
            print(torch.utils.benchmark.Timer("g.replay()", globals={"g": make_graph(1000, func)}).blocked_autorange(min_run_time=1).mean / 1000)