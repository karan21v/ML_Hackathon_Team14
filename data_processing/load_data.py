"""Read PathSnap trip files directly from the supplied student ZIP.

The ZIP is indexed by filename rather than assuming one specific outer directory.
Only the training/student-testing partitions are exposed, never evaluator files.
"""
from __future__ import annotations
from pathlib import Path
import io
import zipfile
import pandas as pd

PARTITIONS = ("train", "student_test")
COLUMNS = {
    "track": ("longitude", "latitude", "timestamp"),
    "nodes": ("longitude", "latitude"),
    "arcs": ("from_node_id", "to_node_id"),
    "route": ("arc_id",),
}


class PathSnapDataset:
    def __init__(self, zip_path: str | Path):
        self.path = Path(zip_path)
        self.zip = zipfile.ZipFile(self.path)
        self.entries: dict[tuple[str, str, str], str] = {}
        for name in self.zip.namelist():
            parts = name.split("/")
            if len(parts) < 3:
                continue
            partition, trip, basename = parts[-3:]
            if partition not in PARTITIONS or "." not in basename:
                continue
            stem, ext = basename.rsplit(".", 1)
            if stem != trip or ext not in COLUMNS:
                continue
            key = (partition, trip, ext)
            if key in self.entries:
                raise ValueError(f"Duplicate trip file for {key}")
            self.entries[key] = name

    def trips(self, partition: str) -> list[str]:
        self._check_partition(partition)
        return sorted({trip for p, trip, _ in self.entries if p == partition})

    @staticmethod
    def _check_partition(partition: str) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"Restricted partition: {partition!r}. Use only {PARTITIONS}")

    def read(self, partition: str, trip: str, ext: str) -> pd.DataFrame:
        self._check_partition(partition)
        if ext not in COLUMNS:
            raise ValueError(f"Unknown file type: {ext}")
        key = partition, trip, ext
        if key not in self.entries:
            raise FileNotFoundError(f"Missing {partition}/{trip}/{trip}.{ext}")
        raw = self.zip.read(self.entries[key])
        if not raw.strip():
            return pd.DataFrame(columns=COLUMNS[ext])
        frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None,
                            names=list(COLUMNS[ext]), on_bad_lines="error")
        for col in COLUMNS[ext]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def file_count(self, partition: str, trip: str, ext: str) -> int | None:
        key = (partition, trip, ext)
        if key not in self.entries:
            return None
        raw = self.zip.read(self.entries[key])
        return len(raw.splitlines())

    def close(self) -> None:
        self.zip.close()

    def __enter__(self) -> "PathSnapDataset":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
