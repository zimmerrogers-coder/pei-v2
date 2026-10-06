"""Import files from the command line:

    .venv\\Scripts\\python.exe -m app.importers <file or folder> [more ...]

TrackMan files go in first, then Dartfish files, because a tag file can only be
attached to a game that is already there.
"""

import sys
from pathlib import Path

from app import db
from app.importers import dartfish, store, trackman


def _collect(arguments: list[str]) -> list[Path]:
    paths = []
    for argument in arguments:
        path = Path(argument)
        paths.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    return paths


def main(arguments: list[str]) -> int:
    if not arguments:
        print(__doc__)
        return 2
    paths = _collect(arguments)
    order = ([p for p in paths if trackman.is_trackman(str(p))]
             + [p for p in paths if dartfish.is_dartfish(str(p))])
    skipped = [p for p in paths if p not in order]

    with db.connect() as conn:
        db.apply_schema(conn)
        for path in order:
            try:
                result = store.import_file(conn, str(path))
            except store.ImportRefused as refusal:
                print(f"REFUSED   {path.name}: {refusal}")
                continue
            if result["source"] == "trackman":
                print(f"trackman  {path.name}: {result['pitches']} pitches, game {result['game_id']}")
            else:
                print(f"dartfish  {path.name}: {result['tags']} tags -> game {result['game_id']}, "
                      f"{len(result['conflicts'])} conflicts with TrackMan")
                for c in result["conflicts"]:
                    print(f"            pitch {c['pitch_no']:>3} {c['hitter']}: {c['field']} "
                          f"Dartfish={c['dartfish']!r} TrackMan={c['trackman']!r}")
    for path in skipped:
        print(f"skipped   {path.name}: not a TrackMan or Dartfish file")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
