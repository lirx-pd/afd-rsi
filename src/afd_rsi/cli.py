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
    if arguments and arguments[0] == "optimize":
        from .optimize import main as optimize_main
        try:
            return optimize_main(arguments[1:], root)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"afd-rsi: {error}", file=sys.stderr)
            return 2
    if arguments and arguments[0] == "native":
        from .native import main as native_main
        try:
            return native_main(arguments[1:], root)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"afd-rsi: {error}", file=sys.stderr)
            return 2
    if arguments and arguments[0] == "service":
        try:
            skills.validate(root)
        except (OSError, ValueError) as error:
            print(f"afd-rsi: {error}", file=sys.stderr)
            return 2
        # The service runner owns signal handling and bounded child cleanup.
        os.execv(sys.executable, [sys.executable, str(root / "benchmarks/service/run.py"), *arguments[1:]])
    parser = argparse.ArgumentParser(prog="afd-rsi")
    parser.epilog = "Local NPU: afd-rsi native --help (build/compare); afd-rsi service --help (plan/run/status/pack)."
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("optimize", help="One-command background optimization; local or iTASK executor")
    check = commands.add_parser("check")
    check.add_argument("recipe", nargs="?", type=Path)
    context = commands.add_parser("context")
    context.add_argument("skill")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("recipe", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("baseline", "compare", "report", "resume"):
        subparser = commands.add_parser(name)
        subparser.add_argument("run", type=Path)
        if name == "compare":
            subparser.add_argument("--operator-only", action="store_true", help="Numerical and paired native timing screen; no service/profile and no promotion")
    search = commands.add_parser("search")
    search.add_argument("run", type=Path)
    search.add_argument("--external-agent", action="store_true", help="exchange request/proposal files with a separate proposer host")
    proposer = commands.add_parser("propose")
    proposer.add_argument("request", type=Path)
    proposer.add_argument("--output", type=Path, required=True)
    patch = commands.add_parser("patch")
    patch.add_argument("run", type=Path)
    patch.add_argument("patch", type=Path)
    export = commands.add_parser("export")
    export.add_argument("run", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--best", action="store_true", help="export the confirmed search incumbent")
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
            print(campaign.run_comparison(args.run.resolve(), root, level="operator" if args.operator_only else "full"))
        elif args.action in ("search", "resume"):
            from .search import search
            run = args.run.resolve()
            if args.action == "search" or (run / "search/state.json").is_file():
                print(search(run, root, external=getattr(args, "external_agent", False)))
                if (run / "search/pending.json").exists():
                    print((run / "search/pending.json").read_text())
            else:
                print(campaign.resume(run, root))
        elif args.action == "propose":
            from .search import propose
            print(propose(args.request.resolve(strict=True), args.output.resolve(), root))
        elif args.action == "report":
            from .search import search_report
            run = args.run.resolve()
            print(search_report(run, root) if (run / "search/state.json").exists() else campaign.report(run, root))
        elif args.action == "export":
            print(campaign.export(args.run.resolve(), args.output.resolve(), root, best=args.best))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"afd-rsi: {error}", file=sys.stderr)
        return 2
