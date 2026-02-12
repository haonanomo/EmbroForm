
import os, time, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="Results folder that Blender uses")
parser.add_argument("--interval", type=float, default=0.5, help="seconds between updates")
args = parser.parse_args()

hl_dir = os.path.join(args.out, "highlight")
os.makedirs(hl_dir, exist_ok=True)
sig_path = os.path.join(hl_dir, "signal.txt")

print(f"Writing numbers to {sig_path} every {args.interval}s.")
try:
    while True:
        val = random.randint(0, 6)
        with open(sig_path, "w", encoding="utf-8") as f:
            f.write(str(val))
        time.sleep(args.interval)
except KeyboardInterrupt:
    print("Stopped.")
