"""Download Citi Bike trips and build the train / test demand models.

    python scripts/prepare_data.py --config configs/default.yaml

Train model: `data.train_months`; held-out test model: `data.test_months`,
built on the same station set so a trained agent can be evaluated on unseen days.
Dock counts come from the live GBFS feed and are cached in
`data/dock_capacity.json` (committed) so results are reproducible.
"""
import argparse
import json
from pathlib import Path

import yaml

from rebike.data import build_demand_model, download_months, fetch_dock_capacity, load_trips


def dock_capacity(path: Path, refresh: bool) -> dict:
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    try:
        caps = fetch_dock_capacity()
    except Exception as e:  # offline: fall back to the throughput estimate
        print(f"[data] could not fetch GBFS dock counts ({e}); using estimates")
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(caps, indent=0, sort_keys=True))
    print(f"[data] cached {len(caps)} dock counts -> {path}")
    return caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--refresh-docks", action="store_true", help="re-download GBFS dock counts")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))["data"]
    raw = Path(cfg["raw_dir"])
    caps = dock_capacity(Path(cfg["dock_capacity"]), args.refresh_docks)

    train_trips = load_trips(download_months(cfg["train_months"], cfg["region"], raw))
    print(f"[data] train: {len(train_trips):,} trips ({', '.join(cfg['train_months'])})")
    train = build_demand_model(train_trips, n_stations=cfg["n_stations"],
                               weekdays_only=cfg["weekdays_only"], dock_capacity=caps)
    train.save(Path(cfg["out_train"]))

    test_trips = load_trips(download_months(cfg["test_months"], cfg["region"], raw))
    print(f"[data] test:  {len(test_trips):,} trips ({', '.join(cfg['test_months'])})")
    test = build_demand_model(test_trips, weekdays_only=cfg["weekdays_only"],
                              station_ids=train.station_ids, dock_capacity=caps)
    test.capacity = train.capacity  # same physical stations
    test.save(Path(cfg["out_test"]))

    for name, m in [("train", train), ("test", test)]:
        print(f"[data] {name}: {len(m.station_ids)} stations, {m.n_trips:,} in-network trips over "
              f"{m.n_days} weekdays, {m.dep_rate.sum():.0f} trips/day, {m.capacity.sum()} docks")


if __name__ == "__main__":
    main()
