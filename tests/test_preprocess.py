import io
import zipfile
import pandas as pd
import pytest
from data_processing.load_data import PathSnapDataset
from data_processing.preprocess import haversine_m, preprocess_track


def test_great_circle_distance():
    assert haversine_m(0, 0, 0, 0) == pytest.approx(0)
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111195, rel=0.001)


def test_preserves_ids_and_flags_time_gap():
    frame = pd.DataFrame({"longitude": [8.0, 8.0, 8.1],
                          "latitude": [49.0, 49.0, 49.1],
                          "timestamp": [1, 2, 8]})
    clean = preprocess_track(frame)
    assert list(clean.obs_idx) == [0, 1, 2]
    assert clean.repeated_position.sum() == 1
    assert clean.large_time_gap.sum() == 1
    assert clean.gps_valid.all()


def test_rejects_reserved_partitions(tmp_path):
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x/train/0001/0001.track", "8 49 1\n")
        z.writestr("x/evaluator_test/0002/0002.track", "9 50 1\n")
    with PathSnapDataset(path) as ds:
        assert ds.trips("train") == ["0001"]
        with pytest.raises(ValueError):
            ds.trips("evaluator_test")
        assert len(ds.read("train", "0001", "track")) == 1
