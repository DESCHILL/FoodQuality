import argparse
import json
from .core import Dataset


def main():
    parser = argparse.ArgumentParser(description="FoodQuality-8 dataset toolkit")
    parser.add_argument("--root", default="workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p = sub.add_parser("taxonomy"); p.add_argument("file")
    p = sub.add_parser("configure"); p.add_argument("file")
    p = sub.add_parser("download"); p.add_argument("source")
    p = sub.add_parser("ingest"); p.add_argument("source"); p.add_argument("directory")
    p = sub.add_parser("deduplicate"); p.add_argument("--threshold", type=int, default=6)
    p = sub.add_parser("review-export"); p.add_argument("--file")
    p = sub.add_parser("review-import"); p.add_argument("file")
    p = sub.add_parser("split"); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("export"); p.add_argument("version"); p.add_argument("--public", action="store_true")
    sub.add_parser("stats")
    sub.add_parser("coverage")
    a = parser.parse_args()
    d = Dataset(a.root)
    try:
        if a.command == "init": result = str(d.root)
        elif a.command == "taxonomy": result = d.set_taxonomy(a.file)
        elif a.command == "configure": result = d.configure(a.file)
        elif a.command == "download": result = d.download(a.source)
        elif a.command == "ingest": result = d.ingest(a.source, a.directory)
        elif a.command == "deduplicate": result = d.deduplicate(a.threshold)
        elif a.command == "review-export": result = d.review_export(a.file)
        elif a.command == "review-import": result = d.review_import(a.file)
        elif a.command == "split": result = d.split(a.seed)
        elif a.command == "export": result = d.export(a.version, a.public)
        elif a.command == "coverage": result = d.coverage()
        else: result = d.statistics()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        d.close()
