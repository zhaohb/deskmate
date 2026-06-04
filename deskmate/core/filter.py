"""Window / app filters.

Patterns:
  * bare token  ->  matches if substring is in app OR title (case-insensitive)
  * "App::Title" -> scoped: app substring AND title substring
"""

from __future__ import annotations

from dataclasses import dataclass


def _norm(s: str) -> str:
    return (s or "").lower()


@dataclass(frozen=True)
class Pattern:
    app: str | None
    title: str

    @classmethod
    def parse(cls, raw: str) -> Pattern:
        if "::" in raw:
            app, title = raw.split("::", 1)
            return cls(app=_norm(app), title=_norm(title))
        return cls(app=None, title=_norm(raw))

    def matches(self, app: str, title: str) -> bool:
        a, t = _norm(app), _norm(title)
        if self.app is not None and self.app not in a:
            return False
        return self.title in t or self.title in a


def is_app_excluded(app_name: str, excluded: list[str]) -> bool:
    a = _norm(app_name)
    return any(token in a for token in (_norm(x) for x in excluded))


class WindowFilter:
    """Combines exclude / include lists. `passes` returns True when window
    should be captured."""

    def __init__(
        self,
        ignored_apps: list[str] | None = None,
        ignored_windows: list[str] | None = None,
        included_windows: list[str] | None = None,
    ) -> None:
        self.ignored_apps = list(ignored_apps or [])
        self.ignored = [Pattern.parse(x) for x in (ignored_windows or [])]
        self.included = [Pattern.parse(x) for x in (included_windows or [])]

    def passes(self, app: str, title: str) -> bool:
        if is_app_excluded(app, self.ignored_apps):
            return False
        if any(p.matches(app, title) for p in self.ignored):
            return False
        if not self.included:
            return True
        scoped_for_app = [p for p in self.included if p.app and p.app in _norm(app)]
        if scoped_for_app:
            return any(p.matches(app, title) for p in scoped_for_app)
        return any(p.matches(app, title) for p in self.included)
