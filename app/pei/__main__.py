"""PEI from the command line:

    .venv\\Scripts\\python.exe -m app.pei all <folder or files>   import, build and fit in one go

    .venv\\Scripts\\python.exe -m app.pei build      rebuild plate appearances and features
    .venv\\Scripts\\python.exe -m app.pei xstats     score the expected-stats model, draw its chart
    .venv\\Scripts\\python.exe -m app.pei reliability  split-half reliability per metric (about 4 minutes)
    .venv\\Scripts\\python.exe -m app.pei fit [--version v2.0] [--notes "..."]   fit the model, save projections
    .venv\\Scripts\\python.exe -m app.pei recovery   test the model on simulated hitters
    .venv\\Scripts\\python.exe -m app.pei evaluate [--version v2.0]   holdout test against baselines (about 20 minutes)
"""

import sys

from app import db
from app.pei import build as build_module


def _build() -> int:
    with db.connect() as conn:
        db.apply_schema(conn)
        result = build_module.build(conn)

    print(f"{result['pitches']} pitches -> {result['plate_appearances']} plate appearances, "
          f"{result['hitters']} hitters, {result['hitter_games']} hitter-games")
    for row in result["problems"].itertuples(index=False):
        print(f"  left out: {row.game_id} inning {row.inning} {row.top_bottom} "
              f"PA {row.pa_of_inning} ({row.pitches} pitches): {row.reason}")

    failed = 0
    for name, rows in result["checks"].items():
        status = "ok  " if rows.empty else "FAIL"
        failed += not rows.empty
        print(f"  {status} {name}" + ("" if rows.empty else f"  ({len(rows)} rows)"))
        if not rows.empty:
            print(rows.head(10).to_string(index=False))
    return 1 if failed else 0


def _xstats() -> int:
    from app.pei import xstats_report
    with db.connect() as conn:
        scores = xstats_report.run(conn)
    print(f"Scored on held-out balls ({scores['balls'].iloc[0]} measured balls, lower is better):")
    for row in scores.itertuples(index=False):
        loss = "   n/a" if row.log_loss != row.log_loss or row.log_loss is None else f"{row.log_loss:.3f}"
        print(f"  {row.method:<34} log loss {loss}   rmse {row.rmse:.3f}")
    print(f"written: {xstats_report.REPORTS / 'xstats_eval.json'}, xstats_surface.png")
    return 0


def _reliability() -> int:
    from app.pei import reliability_report
    with db.connect() as conn:
        db.apply_schema(conn)
        _, summary = reliability_report.run(conn)
    print(f"{'metric':<18}{'per':<21}{'league':>8}{'median n':>10}{'k':>8}{'80% range':>16}"
          f"{'n for r=0.70':>14}")
    for row in summary.itertuples(index=False):
        if row.k == row.k and row.k is not None:
            high = "no signal" if row.k_hi >= 99_000 else f"{row.k_hi:.0f}"
            k, span, stable = f"{row.k:.0f}", f"{row.k_lo:.0f} to {high}", f"{row.stabilization_n:.0f}"
        else:
            k, span, stable = "none", "", ""
        print(f"{row.metric:<18}{row.denominator:<21}{row.league_value:>8.3f}"
              f"{row.median_events:>10.0f}{k:>8}{span:>16}{stable:>14}")
    print(f"written: pei.metric_reliability, pei.reliability_curve, "
          f"{reliability_report.REPORTS / 'reliability.png'}")
    return 0


def _option(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _print_diagnostics(d: dict) -> None:
    print(f"  divergences {d['divergences']} (need 0)   max r-hat {d['max_rhat']:.4f} (need < 1.01)   "
          f"min effective draws {d['min_ess']:.0f} (need > 400)")


def _fit() -> int:
    from app.pei import fit
    with db.connect() as conn:
        db.apply_schema(conn)
        result = fit.run(conn, _option("--version", "v2.0"), _option("--notes", ""))
        names = dict(conn.execute("SELECT player_id, full_name FROM core.player").fetchall())
    print(f"version {result['version']}: {'clean fit' if result['clean'] else 'FIT NOT CLEAN'}")
    _print_diagnostics(result["diagnostics"])
    print("  does simulated data from the model look like the real data?")
    for c in result["checks"]:
        mark = "ok  " if c["inside"] else "OFF "
        print(f"    {mark}{c['check']:<32} real {c['observed']:>8.3f}   "
              f"model {c['model_lo']:.3f} to {c['model_hi']:.3f}")
    print("  league")
    for row in result["league"].itertuples(index=False):
        print(f"    {row.parameter:<14}{row.mean:>8.3f}   80%: {row.lo_80:.3f} to {row.hi_80:.3f}")
    top = result["hitters"].sort_values("n_pa", ascending=False).head(12)
    print(f"  {'hitter':<22}{'PA':>4}{'raw wOBA':>10}{'projected':>11}   80% interval")
    for row in top.itertuples(index=False):
        print(f"  {names[row.batter_id]:<22}{row.n_pa:>4}{row.raw_woba:>10.3f}{row.proj_woba:>11.3f}"
              f"   {row.woba_lo:.3f} to {row.woba_hi:.3f}")
    return 0 if result["clean"] else 1


def _recovery() -> int:
    from app.pei import fit
    result = fit.run_recovery()
    print(f"{len(result['runs'])} simulated leagues of {result['runs'][0]['hitters']} hitters "
          "with known talent")
    print(f"  {'seed':>4}{'raw miss':>10}{'model miss':>12}{'80% cover':>11}{'diverg.':>9}{'r-hat':>8}")
    for run in result["runs"]:
        d = run["diagnostics"]
        print(f"  {run['seed']:>4}{run['rmse_raw']:>10.3f}{run['rmse_model']:>12.3f}"
              f"{run['coverage_80']:>11.0%}{d['divergences']:>9}{d['max_rhat']:>8.4f}")
    average = result["average"]
    print(f"  {'avg':>4}{average['rmse_raw']:>10.3f}{average['rmse_model']:>12.3f}"
          f"{average['coverage_80']:>11.0%}")
    print(f"written: {fit.REPORTS / 'model_recovery.png'}")
    return 0


def _evaluate() -> int:
    from app.pei import evaluate, evaluate_report
    with db.connect() as conn:
        db.apply_schema(conn)
        result = evaluate_report.run(conn, _option("--version", "v2.0"))
    for scheme, part in result["scores"].groupby("scheme"):
        first = part.iloc[0]
        print(f"\n{scheme}: {first['hitters']} hitter-splits, {first['holdout_pa']} held-out PA")
        print(f"  {'method':<26}{'RMSE':>8}{'MAE':>8}{'bias':>8}{'rank':>7}{'80% cover':>11}")
        for row in part.sort_values("rmse").itertuples(index=False):
            rank = "" if row.rank_corr is None or row.rank_corr != row.rank_corr else f"{row.rank_corr:.2f}"
            cover = "" if row.coverage_80 is None or row.coverage_80 != row.coverage_80 else f"{row.coverage_80:.0%}"
            print(f"  {row.method:<26}{row.rmse:>8.4f}{row.mae:>8.4f}{row.bias:>+8.3f}{rank:>7}{cover:>11}")
        print(f"  {evaluate.MODEL} against each other method (positive gain = model better):")
        for row in result["comparisons"][result["comparisons"]["scheme"] == scheme].itertuples(index=False):
            print(f"    vs {row.versus:<26} gain {row.rmse_gain:+.4f}  (80%: {row.gain_lo:+.4f} to "
                  f"{row.gain_hi:+.4f})  better in {row.share_better:.0%} of resamples")
    dirty = [f for f in result["fits"] if f["divergences"] > 0 or f["max_rhat"] >= 1.01]
    print(f"\n{len(result['fits'])} model fits, {len(dirty)} with a divergence or r-hat >= 1.01")
    for f in dirty:
        print(f"  {f['split']}, {f['method']}: divergences {f['divergences']}, r-hat {f['max_rhat']:.4f}")
    print(f"written: pei.evaluation, pei.evaluation_comparison, "
          f"{evaluate_report.REPORTS / 'evaluation.png'}")
    return 0


def _all() -> int:
    """Game files in, projections out: import, build, fit."""
    from app.importers.__main__ import main as import_files
    paths = [a for a in sys.argv[2:] if not a.startswith("--")
             and a not in (_option("--version", ""), _option("--notes", ""))]
    if not paths:
        print("usage: python -m app.pei all <folder or files> [--version v2.0]")
        return 2
    import_files(paths)
    return _build() or _fit()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "all":
        sys.exit(_all())
    if command == "evaluate":
        sys.exit(_evaluate())
    if command == "fit":
        sys.exit(_fit())
    if command == "recovery":
        sys.exit(_recovery())
    if command == "reliability":
        sys.exit(_reliability())
    if command == "build":
        sys.exit(_build())
    if command == "xstats":
        sys.exit(_xstats())
    print(__doc__)
    sys.exit(2)
