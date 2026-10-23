"""Pinned local AFD service lifecycle; executed directly, not imported by control code."""

import os

from afd_e2e.cli import main, parser


def run():
    options = parser().parse_args()
    if options.command in ("plan", "run"):
        if not options.baseline and options.repo is None:
            raise SystemExit("AFD service tests require explicit --repo pointing to the measured checkout")
        if options.command == "run" and not options.devices and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            raise SystemExit("Provide --device or preallocated ASCEND_RT_VISIBLE_DEVICES; no allocation is performed")
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
