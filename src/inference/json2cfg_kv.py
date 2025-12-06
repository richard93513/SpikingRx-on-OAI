import json
import sys

if len(sys.argv) != 3:
    print("Usage: json2cfg_kv.py ldpc_cfg.json output_cfg.txt")
    exit(1)

with open(sys.argv[1]) as f:
    cfg = json.load(f)

with open(sys.argv[2], "w") as f:
    for k, v in cfg.items():
        f.write(f"{k} {v}\n")

print("Wrote", sys.argv[2])

