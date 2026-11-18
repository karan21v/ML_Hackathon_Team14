"""Synthetic miniature numeric-trip ZIP ensures the parser never uses .route."""
import io
import numpy as np
import pytest
import zipfile
from src.api.real_trips import DatasetTripZip, RealTripMatcher, RealMatchOptions, FEATURES


class FixtureModel:
    def predict_proba(self,X):
        x=np.asarray(X,dtype=float)
        assert x.shape[1]==8
        # A toy score function, NOT an actual evaluation metric.
        p=1/(1+np.exp(np.minimum(x[:,0],200)/25-2))
        return np.column_stack((1-p,p))


def create_fixture(path,partition='student_test'):
    lon0,lat0=43.68,49.77
    nodes=np.array([[lon0+i*.00025,lat0] for i in range(9)])
    arcs=np.array([[i,i+1] for i in range(8)]+[[i+1,i] for i in range(8)])
    track=np.array([[lon0+(.5+i)*.00025,lat0+(.000003 if i%2 else -.000003),i+1]
                    for i in range(7)])
    buf={}
    for ext, data, fmt in [('nodes',nodes,'%.8f'),('arcs',arcs,'%d'),('track',track,'%.8f')]:
        b=io.StringIO();np.savetxt(b,data,fmt=fmt);buf[ext]=b.getvalue()
    with zipfile.ZipFile(path,'w') as z:
        prefix=f'{partition}/00000042/00000042'
        for ext, raw in buf.items():z.writestr(prefix+f'.{ext}',raw)
        z.writestr(prefix+'.route','REFERENCE MUST NOT BE READ!')
    return track


def test_real_numeric_trip_is_inferred_without_reference(tmp_path):
    path=tmp_path/'student.zip';track=create_fixture(path)
    ds=DatasetTripZip(path)
    assert ds.trip_ids()==['00000042']
    matcher=RealTripMatcher({'feature_names':list(FEATURES),'model_name':'fixture',
                            'model':FixtureModel()})
    r=matcher.match(ds,'00000042',RealMatchOptions(max_points=5))
    p=r['prediction']
    assert p['num_points']==5 and p['total_trip_points']==7
    assert p['preview_only'] is True
    assert p['accuracy'] is None and p['ml_only_accuracy'] is None
    assert len(p['snapped_points'])==5
    assert all(0<=int(x)<16 for x in p['matched_edges_sequence'])
    assert 'Selected candidate road arcs' in r['map_html']
    assert ds.details('00000042')['dataset_type']=='student_test'
    ds.close()


def test_real_eval_partition_locked_by_default(tmp_path,monkeypatch):
    path=tmp_path/'evaluator.zip';create_fixture(path,'evaluator_test')
    monkeypatch.delenv('PATHSNAP_ALLOW_EVALUATOR',raising=False)
    with pytest.raises(ValueError,match='locked'):
        DatasetTripZip(path,'evaluator_test')


def test_unavailable_trip_fails_without_exposing_arbitrary_files(tmp_path):
    path=tmp_path/'student.zip';create_fixture(path)
    ds=DatasetTripZip(path)
    with pytest.raises(KeyError):ds.read('../bad','track')
    ds.close()
