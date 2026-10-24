"""Citi Bike trip data -> demand model for the ReBike simulator.

Pipeline:
    1. download monthly trip archives from the public Citi Bike S3 bucket
    2. load trips, keep the N busiest stations (both trip ends inside the set)
    3. estimate, per hour-of-day h:
         dep_rate[h, i]   expected departures per hour from station i
         od_prob[h, i, j] probability that a trip leaving i ends at j
       plus station coordinates, a ride-time matrix and a capacity estimate
    4. save everything to a single .npz consumed by `RebalancingEnv`
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TRIPDATA_URL = "https://s3.amazonaws.com/tripdata/{name}"
GBFS_STATION_INFO = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
USECOLS = [
    "started_at", "ended_at",
    "start_station_id", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
]


# --------------------------------------------------------------------------- #
# download / load
# --------------------------------------------------------------------------- #
def archive_name(month: str, region: str = "JC") -> str:
    """month='202409'. region 'JC' (Jersey City/Hoboken) or 'NYC'."""
    if region.upper() == "JC":
        return f"JC-{month}-citibike-tripdata.csv.zip"
    return f"{month}-citibike-tripdata.zip"


def download_months(months: list[str], region: str, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in months:
        name = archive_name(m, region)
        dest = raw_dir / name
        if not dest.exists():
            print(f"[data] downloading {name}")
            r = requests.get(TRIPDATA_URL.format(name=name), timeout=600)
            r.raise_for_status()
            dest.write_bytes(r.content)
        paths.append(dest)
    return paths


def _read_zip(path: Path) -> pd.DataFrame:
    """Read every trip CSV inside a (possibly nested) Citi Bike zip."""
    frames = []

    def _walk(zf: zipfile.ZipFile):
        for info in zf.infolist():
            n = info.filename
            if n.startswith("__MACOSX") or n.endswith("/"):
                continue
            if n.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                    _walk(inner)
            elif n.endswith(".csv"):
                with zf.open(info) as f:
                    frames.append(pd.read_csv(f, usecols=USECOLS, low_memory=False))

    with zipfile.ZipFile(path) as zf:
        _walk(zf)
    return pd.concat(frames, ignore_index=True)


def load_trips(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([_read_zip(p) for p in paths], ignore_index=True)
    df = df.dropna(subset=["start_station_id", "end_station_id"])
    df["start_station_id"] = df["start_station_id"].astype(str)
    df["end_station_id"] = df["end_station_id"].astype(str)
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")
    df["ended_at"] = pd.to_datetime(df["ended_at"], format="mixed")
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    # drop obviously broken rides
    df = df[(df["duration_min"] > 1) & (df["duration_min"] < 180)]
    return df


# --------------------------------------------------------------------------- #
# demand model
# --------------------------------------------------------------------------- #
@dataclass
class DemandModel:
    station_ids: np.ndarray      # (N,)
    coords: np.ndarray           # (N, 2) lat, lng
    capacity: np.ndarray         # (N,)  docks
    dep_rate: np.ndarray         # (24, N) departures / hour
    od_prob: np.ndarray          # (24, N, N)
    ride_min: np.ndarray         # (N, N) median ride time in minutes
    dist_km: np.ndarray          # (N, N) road-distance estimate
    n_trips: int
    n_days: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "DemandModel":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in cls.__dataclass_fields__}
        kw["n_trips"] = int(kw["n_trips"])
        kw["n_days"] = int(kw["n_days"])
        return cls(**kw)


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(a[:, None, 0]), np.radians(a[:, None, 1])
    lat2, lon2 = np.radians(b[None, :, 0]), np.radians(b[None, :, 1])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def fetch_dock_capacity(min_docks: int = 5) -> dict[str, int]:
    """Real dock counts from the live Citi Bike GBFS feed, keyed by trip-data station id.

    Trip archives use the GBFS `short_name` (e.g. 'JC115', 'HB102'). Stations
    reporting fewer than `min_docks` docks (inactive / being serviced) are skipped.
    """
    r = requests.get(GBFS_STATION_INFO, timeout=60)
    r.raise_for_status()
    out = {}
    for s in r.json()["data"]["stations"]:
        sid, cap = str(s.get("short_name", "")), int(s.get("capacity") or 0)
        if sid and cap >= min_docks:
            out[sid] = cap
    return out


def build_demand_model(
    df: pd.DataFrame,
    n_stations: int = 30,
    weekdays_only: bool = True,
    detour_factor: float = 1.35,
    od_smoothing: float = 0.05,
    station_ids: np.ndarray | None = None,
    dock_capacity: dict[str, int] | None = None,
) -> DemandModel:
    """Estimate the demand model.

    station_ids   reuse a fixed station set (e.g. the training set when building
                  the held-out test model); default = the `n_stations` busiest.
    dock_capacity real dock counts {station_id: docks}; stations missing from it
                  fall back to a throughput-based estimate.
    """
    if weekdays_only:
        df = df[df["started_at"].dt.dayofweek < 5]

    if station_ids is None:
        # busiest stations by total activity
        activity = pd.concat([df["start_station_id"], df["end_station_id"]]).value_counts()
        ids = activity.index[:n_stations].to_numpy()
    else:
        ids = np.asarray(station_ids)
    idx = {s: k for k, s in enumerate(ids)}
    df = df[df["start_station_id"].isin(idx) & df["end_station_id"].isin(idx)].copy()
    df["i"] = df["start_station_id"].map(idx)
    df["j"] = df["end_station_id"].map(idx)
    df["h"] = df["started_at"].dt.hour
    n_days = df["started_at"].dt.normalize().nunique()
    N = len(ids)

    # coordinates: median of reported start positions (fall back to end positions)
    start = df.groupby("i")[["start_lat", "start_lng"]].median().reindex(range(N))
    end = df.groupby("j")[["end_lat", "end_lng"]].median().reindex(range(N))
    coords = start.to_numpy().copy()
    miss = np.isnan(coords[:, 0])
    coords[miss] = end.to_numpy()[miss]

    # departure rates
    counts = np.zeros((24, N))
    np.add.at(counts, (df["h"].to_numpy(), df["i"].to_numpy()), 1)
    dep_rate = counts / max(n_days, 1)

    # origin-destination probabilities (hourly, with smoothing toward daily OD)
    od = np.zeros((24, N, N))
    np.add.at(od, (df["h"].to_numpy(), df["i"].to_numpy(), df["j"].to_numpy()), 1)
    daily = od.sum(0, keepdims=True)
    daily = daily / np.clip(daily.sum(-1, keepdims=True), 1, None)
    hourly = od / np.clip(od.sum(-1, keepdims=True), 1, None)
    od_prob = (1 - od_smoothing) * hourly + od_smoothing * daily
    empty = od.sum(-1) == 0
    od_prob[empty] = daily[0][np.where(empty)[1]]
    od_prob /= np.clip(od_prob.sum(-1, keepdims=True), 1e-12, None)

    # ride time: median per OD pair, fall back to distance / 12 km/h
    dist_km = haversine_km(coords, coords) * detour_factor
    ride_min = dist_km / 12.0 * 60 + 2.0
    med = df.groupby(["i", "j"])["duration_min"].median()
    ride_min[med.index.get_level_values(0), med.index.get_level_values(1)] = med.to_numpy()

    # capacity heuristic: scale with peak hourly throughput, clipped to typical dock counts
    arr = np.zeros((24, N))
    np.add.at(arr, (df["h"].to_numpy(), df["j"].to_numpy()), 1)
    peak = np.maximum(dep_rate, arr / max(n_days, 1)).max(0)
    capacity = np.clip(np.round(15 + 2.0 * peak), 15, 45).astype(int)
    if dock_capacity:
        real = np.array([dock_capacity.get(s, -1) for s in ids])
        capacity = np.where(real > 0, real, capacity).astype(int)
        print(f"[data] real dock counts for {(real > 0).sum()}/{N} stations")

    return DemandModel(
        station_ids=ids.astype(str), coords=coords, capacity=capacity,
        dep_rate=dep_rate, od_prob=od_prob, ride_min=ride_min, dist_km=dist_km,
        n_trips=len(df), n_days=int(n_days),
    )
