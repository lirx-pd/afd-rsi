"""CLI for the first local, manual-candidate implementation slice."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import campaign, skills
from .contracts import load_recipe


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2]
    if arguments and arguments[0] == "service":
        try:
            skills.validate(root)
        except (OSError, ValueError) as error:
            print(f"afd-rsi: {error}", file=sys.stderr)
            return 2
        # The service runner owns signal handling and bounded child cleanup.
        os.execv(sys.executable, [sys.executable, str(root / "benchmarks/service/run.py"), *arguments[1:]])
    parser = argparse.ArgumentParser(prog="afd-rsi")
    parser.epilog = "Real local service tests: afd-rsi service --help (plan/run/status/pack)."
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check")
    check.add_argument("recipe", nargs="?", type=Path)
    context = commands.add_parser("context")
    context.add_argument("skill")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("recipe", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("baseline", "compare", "report"):
        commands.add_parser(name).add_argument("run", type=Path)
    patch = commands.add_parser("patch")
    patch.add_argument("run", type=Path)
    patch.add_argument("patch", type=Path)
    export = commands.add_parser("export")
    export.add_argument("run", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        if args.action == "check":
            manifest = skills.validate(root)
            if args.recipe:
                plan = load_recipe(args.recipe)
                print(json.dumps({"backend": plan["backend"], "source": plan["source"]}))
            print(f"Foundation valid: {len(manifest['skills'])} skills, {len(manifest['files'])} files; hardware unqualified")
        elif args.action == "context":
            print(skills.context(root, args.skill))
        elif args.action == "prepare":
            print(campaign.prepare(args.recipe.resolve(), args.output, root))
        elif args.action == "patch":
            run = args.run.resolve()
            campaign.apply_patch(run, args.patch.resolve(strict=True), root)
            print(run / "workspaces/candidate")
        elif args.action == "baseline":
            print(campaign.run_baseline(args.run.resolve(), root))
        elif args.action == "compare":
            print(campaign.run_comparison(args.run.resolve(), root))
        elif args.action == "report":
            print(campaign.report(args.run.resolve(), root))
        elif args.action == "export":
            print(campaign.export(args.run.resolve(), args.output.resolve(), root))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"afd-rsi: {error}", file=sys.stderr)
        return 2
