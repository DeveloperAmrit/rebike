"""Download Citi Bike trips and build the demand model used by the simulator.

    python scripts/prepare_data.py --config configs/default.yaml
"""
import argparse
from pathlib import Path

import yaml

from rebike.data import build_demand_model, download_months, load_trips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))["data"]

    paths = download_months(cfg["months"], cfg["region"], Path(cfg["raw_dir"]))
    trips = load_trips(paths)
    print(f"[data] loaded {len(trips):,} trips from {len(paths)} archive(s)")

    model = build_demand_model(trips, n_stations=cfg["n_stations"], weekdays_only=cfg["weekdays_only"])
    model.save(Path(cfg["out"]))
    print(f"[data] {len(model.station_ids)} stations, {model.n_trips:,} in-network trips "
          f"over {model.n_days} days -> {cfg['out']}")
    print(f"[data] mean weekday demand: {model.dep_rate.sum():.0f} trips/day, "
          f"total docks: {model.capacity.sum()}")


if __name__ == "__main__":
    main()
