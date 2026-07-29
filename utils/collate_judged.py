"""Collate judge-only sweep results into one comparison table."""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/judged")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        if f.endswith("_dialogues.jsonl"):
            continue
        try:
            rows.append(json.load(open(f))["summary"])
        except Exception:
            continue
    if not rows:
        print(f"no results in {args.dir}")
        return

    hdr = (f"{'run':<24}{'engage':>8}{'learn':>8}{'beh':>6}{'aff':>6}{'cog':>6}"
           f"{'evid':>6}{'diseng':>8}{'msgs':>7}{'leak':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["engagement"]):
        print(f"{r['run_name']:<24}{r['engagement']:>8.3f}{r['learning']:>8.3f}"
              f"{r['behavioral']:>6.2f}{r['affective']:>6.2f}{r['cognitive']:>6.2f}"
              f"{r['learning_evidence']:>6.2f}{r['disengage_rate']:>8.3f}"
              f"{r['avg_dialogue_msgs']:>7.1f}{r['leak_flag_rate']:>7.3f}")

    if args.csv:
        import csv
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
