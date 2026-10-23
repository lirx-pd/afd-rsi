import signal

from .cli import main


def stop(signum, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)

raise SystemExit(main())
