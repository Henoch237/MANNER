"""
launcher.py - SESSION launcher in the terminal (continuous conversation).

Run from the project root:
  export DASHSCOPE_API_KEY=...
  python -m dmp_assistant.launcher

The web front-end with face authentication is dmp_assistant.web.server;
both drive the same dmp_assistant.session.Session.
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmp_assistant.planning.planner import Planner
from dmp_assistant.execution.executor import Executor
from dmp_assistant.session import HELP, Session

ROBOT_IP = "192.168.58.4"       # real robot (sim: 192.168.85.130)


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set. Run: export DASHSCOPE_API_KEY=<your key>")

    print("=" * 60)
    print("LLM PLANNER + EXECUTOR - SESSION MODE")
    print("=" * 60)
    print(HELP)

    session = Session(Planner(), lambda: Executor(ROBOT_IP))

    while True:
        try:
            req = input("\nSession > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            break
        if not req:
            continue
        reply = session.handle(req)
        if reply.kind == "quit":
            print("Session closed.")
            break
        if reply.kind != "plan":
            continue
        confirm = input("\nExecute on robot? (y/n): ").strip().lower()
        if confirm == "y":
            session.execute()
        else:
            session.discard()

    print("\n" + "=" * 60)
    print("SESSION END")
    print("=" * 60)


if __name__ == "__main__":
    main()
