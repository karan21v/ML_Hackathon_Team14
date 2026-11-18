"""Small synthetic tests, no evaluator data or fabricated predicted results."""
from pathlib import Path
import zipfile
import json

import pytest

from data_processing.load_data import PathSnapDataset
from data_processing.compare_routes import make_comparison_map


@pytest.fixture
def synthetic_zip(tmp_path):
    path = tmp_path / 'synthetic.zip'
    base = 'folder/train/00000001/00000001.'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(base+'track', '8.0 49.0 1\n8.001 49.0 2\n8.002 49.0 3\n')
        z.writestr(base+'nodes', '8.0 49.0\n8.001 49.0\n8.002 49.0\n8.003 49.0\n')
        z.writestr(base+'arcs', '0 1\n1 2\n2 3\n')
        z.writestr(base+'route', '0\n1\n2\n')
    return path


def test_reference_only_does_not_claim_predictions(synthetic_zip, tmp_path):
    with PathSnapDataset(synthetic_zip) as ds:
        result = make_comparison_map(ds, 'train', '00000001', tmp_path/'out')
    assert result['predictions_supplied'] is False
    assert result['predicted_observations'] == 0
    html = Path(result['map']).read_text(encoding='utf-8')
    assert 'Predicted route not supplied yet' in html
    assert 'Reference route' in html
    assert json.loads(Path(result['report']).read_text())['reference_arc_entries'] == 3


def test_supplied_prediction_file_draws_red_layer(synthetic_zip,tmp_path):
    predictions = tmp_path/'pred.csv'
    predictions.write_text('trip_id,observation_index,arc_id\n00000001,0,0\n00000001,1,1\n00000001,2,2\n')
    with PathSnapDataset(synthetic_zip) as ds:
        result = make_comparison_map(ds,'train','00000001',tmp_path/'out',predictions)
    assert result['predictions_supplied'] is True
    assert result['predicted_observations'] == 3
    assert result['direct_endpoint_diagnostic']['pairs_examined'] == 2
    assert result['direct_endpoint_diagnostic']['non_touching_pairs'] == 0
    html = Path(result['map']).read_text(encoding='utf-8')
    assert 'Predicted route (decoded)' in html
    assert '#D94841' in html


def test_invalid_arc_rejected(synthetic_zip, tmp_path):
    predictions = tmp_path/'pred.csv'
    predictions.write_text('observation_index,arc_id\n0,0\n1,987654\n')
    with PathSnapDataset(synthetic_zip) as ds:
        with pytest.raises(ValueError, match='out-of-range'):
            make_comparison_map(ds,'train','00000001',tmp_path/'out',predictions)


def test_candidate_rows_rejected(synthetic_zip, tmp_path):
    predictions = tmp_path/'pred.csv'
    predictions.write_text('observation_index,arc_id\n0,0\n0,1\n')
    with PathSnapDataset(synthetic_zip) as ds:
        with pytest.raises(ValueError, match='Duplicate observation_index'):
            make_comparison_map(ds,'train','00000001',tmp_path/'out',predictions)


def test_missing_and_out_of_order_predictions(synthetic_zip,tmp_path):
    predictions = tmp_path/'pred.csv'
    predictions.write_text('observation_index,arc_id\n2,2\n0,0\n')
    with PathSnapDataset(synthetic_zip) as ds:
        result = make_comparison_map(ds,'train','00000001',tmp_path/'out',predictions)
    assert result['predicted_observations'] == 2
    assert result['predicted_observation_fraction'] == pytest.approx(2/3)
    assert result['direct_endpoint_diagnostic']['pairs_examined'] == 0
