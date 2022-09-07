#!/usr/bin/env python
import torch
from tqdm import tqdm

ts = []
with tqdm(desc="Bottleneck") as pbar:
    while True:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        pbar.set_postfix({"Total": t, "Free": f, "Used": a})

        try:
            ts.append(
                torch.empty(
                    1024 * 1024 * 1024 * 8 // 32,
                ).cuda()
            )
        except RuntimeError:
            pass
