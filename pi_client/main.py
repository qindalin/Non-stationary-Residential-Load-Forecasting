from __future__ import annotations

import argparse

from client_runtime.agent import EdgeAgent
from client_runtime.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Raspberry Pi client")
    parser.add_argument("--config", required=True, help="Path to the client YAML config")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for online replay steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    agent = EdgeAgent(config)
    agent.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()