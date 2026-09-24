"""Halo generator command line.

  python halo_cli.py                          # one random halo -> halo_<seed>.svg and .png
  python halo_cli.py --seed 12345             # a specific halo (any text works: --seed tralala)
  python halo_cli.py --count 20 -o out/       # 20 random halos into a folder
  python halo_cli.py --seed 7 --svg-only --size 1024 --color "#ffcc66" --transparent
  python halo_cli.py --seed 7 --info          # print what the halo is made of (JSON)
  python halo_cli.py --issue alice bob        # permanent unique halos per person (registry file)
"""
import argparse
import json
import os
import sys

from halo import api


def hex_to_rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def write(rec, args, stem):
    os.makedirs(args.out, exist_ok=True)
    paths = []
    if not args.png_only:
        p = os.path.join(args.out, stem + ".svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(api.svg(rec, size=args.size, color=args.color, background=None if args.transparent else args.background))
        paths.append(p)
    if not args.svg_only:
        p = os.path.join(args.out, stem + ".png")
        api.png(rec, p, size=args.size, color=args.color, background=hex_to_rgb(args.background))
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate Blue Archive-style halos (SVG / PNG).")
    ap.add_argument("--seed", action="append", help="halo seed, any text (repeatable); default: random")
    ap.add_argument("--count", type=int, default=1, help="number of random halos when no --seed is given")
    ap.add_argument("-o", "--out", default=".", help="output folder (default: current folder)")
    ap.add_argument("--size", type=int, default=512, help="image size in pixels (default 512)")
    ap.add_argument("--color", help='halo color instead of the seed color: "#ff8800", "ff8800" or "#f80"')
    ap.add_argument("--background", default="#1e1e24", help="background color (default #1e1e24)")
    ap.add_argument("--transparent", action="store_true", help="transparent SVG background")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--svg-only", action="store_true")
    fmt.add_argument("--png-only", action="store_true")
    ap.add_argument("--info", action="store_true", help="print the halo's composition as JSON instead of files")
    ap.add_argument("--issue", nargs="+", metavar="PERSON", help="issue permanent unique halos to these people")
    ap.add_argument("--registry", default="registry.jsonl", help="registry file for --issue (default registry.jsonl)")
    args = ap.parse_args(argv)
    try:
        args.color = api.normalize_color(args.color)
        hex_to_rgb(api.normalize_color(args.background))
    except ValueError as e:
        ap.error(str(e))

    if args.issue:
        from halo.build import build
        from halo.issue import Issuer
        iss = Issuer(args.registry)
        for person in args.issue:
            rec = iss.issue(person)
            rec = {**rec, "seed": rec["seed"], "geometry": build(rec["halo"])}
            for p in write(rec, args, f"halo_{api.safe_name(person)}"):
                print(p)
        return 0

    seeds = args.seed or [api.random_seed() for _ in range(args.count)]
    for s in seeds:
        rec = api.halo_for(s)
        if args.info:
            print(json.dumps({k: rec[k] for k in ("seed", "type", "color", "generator", "desc")}, ensure_ascii=False))
            continue
        for p in write(rec, args, f"halo_{api.safe_name(s)}"):
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
