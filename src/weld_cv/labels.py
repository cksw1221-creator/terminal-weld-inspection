from pathlib import Path


def expected_label(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()

    if name.startswith("ok") or "ok" in parts:
        return "ok"
    if "missing" in name:
        return "missing"
    if "less" in name:
        return "less"
    if "ng" in parts:
        return "ng"
    return "unknown"

