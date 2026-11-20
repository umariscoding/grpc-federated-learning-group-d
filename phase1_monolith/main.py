"""Entry point of the monolith.

    python3 main.py            run the scripted demo (A=10, B=20, C=30 -> 20)
    python3 main.py shell      interactive admin console
"""
import argparse
import logging
import sys

from fl_monolith import AdminConsole, FLServer


def run_demo(console: AdminConsole, rounds: int, lr: float) -> None:
    script = [
        "register A 10",
        "register B 20",
        "register C 30",
        "nodes",
        f"create rounds={rounds} lr={lr}",
        "start exp-1",
        "status exp-1",
    ]
    for line in script:
        print(f"fl> {line}")
        console.onecmd(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monolithic Federated Learning prototype")
    parser.add_argument("mode", nargs="?", choices=["demo", "shell"], default="demo")
    parser.add_argument("--rounds", type=int, default=1, help="demo only")
    parser.add_argument("--lr", type=float, default=1.0, help="demo only: local learning rate")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show internal server activity (always on in demo)")
    args = parser.parse_args()

    level = logging.INFO if (args.verbose or args.mode == "demo") else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stdout, format="    [%(name)s] %(message)s")

    console = AdminConsole(FLServer())
    if args.mode == "demo":
        run_demo(console, args.rounds, args.lr)
    else:
        console.cmdloop()


if __name__ == "__main__":
    main()
