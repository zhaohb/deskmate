"""Markdown + YAML-frontmatter pipe loader.

Frontmatter shape:

```
---
name: my-pipe
description: ...
schedule: "*/15 * * * *"          # cron
interval_seconds: 900             # OR this
permissions:
  read_db: true
  trigger_capture: false
  call_llm: false
runtime: python                   # python | js | none
---

# pipe body (markdown / source)
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logger import get

logger = get("pipes.loader")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class PipePermissions:
    read_db: bool = True
    write_db: bool = False
    trigger_capture: bool = False
    call_llm: bool = False
    read_filesystem: bool = False
    write_filesystem: bool = False


@dataclass
class PipeFrontmatter:
    name: str
    description: str = ""
    schedule: str | None = None
    interval_seconds: int | None = None
    runtime: str = "none"  # python | js | none
    permissions: PipePermissions = field(default_factory=PipePermissions)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipe:
    path: Path
    frontmatter: PipeFrontmatter
    body: str


def load_pipes(directory: Path) -> list[Pipe]:
    if not directory.exists():
        return []
    out: list[Pipe] = []
    for md in sorted(directory.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
            fm_match = _FRONTMATTER_RE.match(text)
            if not fm_match:
                logger.debug("pipe %s has no frontmatter; skipping", md.name)
                continue
            fm_dict = _parse_yaml_subset(fm_match.group(1))
            body = text[fm_match.end():]
            out.append(Pipe(path=md, frontmatter=_to_frontmatter(fm_dict), body=body))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse pipe %s: %s", md.name, exc)
    return out


def _to_frontmatter(d: dict[str, Any]) -> PipeFrontmatter:
    perms_raw = d.get("permissions", {}) or {}
    perms = PipePermissions(
        read_db=bool(perms_raw.get("read_db", True)),
        write_db=bool(perms_raw.get("write_db", False)),
        trigger_capture=bool(perms_raw.get("trigger_capture", False)),
        call_llm=bool(perms_raw.get("call_llm", False)),
        read_filesystem=bool(perms_raw.get("read_filesystem", False)),
        write_filesystem=bool(perms_raw.get("write_filesystem", False)),
    )
    interval = d.get("interval_seconds")
    return PipeFrontmatter(
        name=str(d.get("name", "")),
        description=str(d.get("description", "")),
        schedule=d.get("schedule"),
        interval_seconds=int(interval) if interval is not None else None,
        runtime=str(d.get("runtime", "none")),
        permissions=perms,
        raw=d,
    )


# ─── tiny YAML subset parser ───────────────────────────────────────────────
def _parse_yaml_subset(yaml_text: str) -> dict[str, Any]:
    """Just enough to handle local pipe frontmatter:
    - `key: value` (strings, ints, bools, null)
    - nested `key:` followed by indented `  subkey: value` blocks
    - lists with `- item` (rarely used here)

    Pulls in PyYAML if available (round-trip safe), otherwise uses this
    inline parser.
    """
    try:
        import yaml  # type: ignore[import-not-found]
        return yaml.safe_load(yaml_text) or {}
    except Exception:  # noqa: BLE001
        pass

    out: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, out)]
    for raw_line in yaml_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        while stack and indent < stack[-1][0]:
            stack.pop()
        line = raw_line.strip()
        if line.startswith("- "):
            continue  # lists are not used by the default pipe frontmatter
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            new_dict: dict[str, Any] = {}
            stack[-1][1][key] = new_dict
            stack.append((indent + 2, new_dict))
        else:
            stack[-1][1][key] = _coerce(val)
    return out


def _coerce(val: str) -> Any:
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    if val.lower() in ("null", "none", "~"):
        return None
    if val.startswith(('"', "'")) and val[-1] == val[0]:
        return val[1:-1]
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val
