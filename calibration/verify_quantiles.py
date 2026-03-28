import csv
import math

def quantile(xs, q):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        raise ValueError("empty")
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]

    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


clips = []
aess = []

with open("calibration/calibration_scores.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        clips.append(float(row["clip_score"]))
        aess.append(float(row["aesthetic_score"]))

print("rows:", len(clips))
print("clip q0.1:", quantile(clips, 0.1))
print("aes  q0.1:", quantile(aess, 0.1))