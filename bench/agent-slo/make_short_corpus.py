import json
import sys

n = int(sys.argv[1])
out = sys.argv[2]
with open(out, "w") as f:
    for i in range(n):
        json.dump({"prompt": f"short decode test prompt number {i}. reply briefly."}, f)
        f.write("\n")
print("wrote", n, "to", out)
