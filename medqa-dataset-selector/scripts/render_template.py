#!/usr/bin/env python3
"""
Render Unsloth train script templates by substituting {PLACEHOLDER} tokens.

Usage:
    python scripts/render_template.py train_sft.py --model unsloth/Qwen3.5-9B --lora_r 16
    python scripts/render_template.py train_grpo.py --model unsloth/Qwen3.5-27B --lora_r 32 --max_seq_length 4096

Any --key value pair on the CLI is substituted as {KEY} → value in the template.
"""
import argparse
import re
import sys
from pathlib import Path


def find_placeholders(text: str) -> list[str]:
    return re.findall(r"\{([A-Z_]+)\}", text)


def render(template_path: Path, substitutions: dict[str, str]) -> str:
    text = template_path.read_text()
    remaining = find_placeholders(text)
    missing = [p for p in remaining if p not in substitutions]
    if missing:
        print(f"WARNING: unresolved placeholders: {missing}", file=sys.stderr)
    for key, value in substitutions.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Render train script templates")
    parser.add_argument("template", help="Path to template .py file")
    parser.add_argument("--out", help="Output path (default: overwrite in-place)")
    args, extra = parser.parse_known_args()

    substitutions: dict[str, str] = {}
    it = iter(extra)
    for token in it:
        if token.startswith("--"):
            key = token.lstrip("-").upper()
            try:
                substitutions[key] = next(it)
            except StopIteration:
                print(f"ERROR: --{token.lstrip('-')} requires a value", file=sys.stderr)
                sys.exit(1)

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    rendered = render(template_path, substitutions)

    out_path = Path(args.out) if args.out else template_path
    out_path.write_text(rendered)
    print(f"Rendered → {out_path}")

    remaining = find_placeholders(rendered)
    if remaining:
        print(f"WARNING: {len(remaining)} placeholder(s) still unresolved: {remaining}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
