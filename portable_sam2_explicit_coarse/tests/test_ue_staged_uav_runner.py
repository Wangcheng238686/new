"""CPU integration bindings, not a substitute for a real four-rank gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
WORK=Path(__file__).resolve().parents[1]


@pytest.fixture
def run():
    spec=importlib.util.spec_from_file_location('staged_runner_test',WORK/'scripts/probes/ue_staged_uav_run_20261028.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_shared_accumulation_uses_SUTC_objective_instead_of_old_unconditional_aux(run,monkeypatch):
    protocol=run.read(WORK/'configs/ablations/ue_staged_uav_protocol_20261028.json')
    # All process-local substitutions are restored at test exit.
    monkeypatch.setattr(run.full,'PROTOCOL',run.full.PROTOCOL)
    monkeypatch.setattr(run.engine,'PROTOCOL',run.engine.PROTOCOL)
    monkeypatch.setattr(run.previous,'objective',run.previous.objective)
    monkeypatch.setattr(run.engine,'source_inventory',run.engine.source_inventory)
    monkeypatch.setattr(run.engine,'verify_sources',run.engine.verify_sources)
    monkeypatch.setattr(run.full.native,'evaluate',run.full.native.evaluate)
    run.configure({'protocol':protocol,'source_sha256':{}})
    rt=SimpleNamespace(objective=Mock(return_value='correct_SUTC_loss'))
    assert run.previous.objective(rt,'sample',12)=='correct_SUTC_loss'
    rt.objective.assert_called_once_with('sample',12)
    assert run.full.native.evaluate is run.evaluate


def test_gate_identity_requires_all_rank_records_and_exact_recipe(run,tmp_path):
    r={'name':'T','recipe_sha256':'recipe'}
    m={'protocol_sha256':'protocol','source_sha256':{'a.py':'bytes'}}
    path=tmp_path/'gates/T/complete.json'
    value=dict(passed=True,**run.marker_identity(m,r),ranks=[{'passed':True}]*4)
    run.dump(path,value)
    assert run.require_gate(tmp_path,m,r)[0]['passed']
    value['ranks']=value['ranks'][:3];run.dump(path,value)
    with pytest.raises(RuntimeError,match='rank coverage'):run.require_gate(tmp_path,m,r)
    value['ranks']=[{'passed':True}]*4;value['recipe_sha256']='other';run.dump(path,value)
    with pytest.raises(RuntimeError,match='Matching passed'):run.require_gate(tmp_path,m,r)


def test_structure_test_cannot_run_before_explicit_validation_base_lock(run,monkeypatch,tmp_path):
    m={'base_lock':None,'stages':{'structure':['S','U','T','C']}}
    monkeypatch.setattr(run,'validate_manifest',Mock(return_value=(m,{'name':'C'})))
    monkeypatch.setattr(run,'configure',Mock())
    setup=Mock(side_effect=AssertionError('GPU setup must not run'))
    monkeypatch.setattr(run.engine,'setup',setup)
    monkeypatch.setattr(run.sys,'argv',['run','eval','--root',str(tmp_path),'--recipe','C'])
    with pytest.raises(RuntimeError,match='before exposing'):run.main()
    setup.assert_not_called()


def test_direct_runner_rejects_missing_recipe_cache_receipt(run,monkeypatch,tmp_path):
    source=tmp_path/'source';work=source/'package';root=tmp_path/'study'
    root.mkdir();work.mkdir(parents=True)
    matrix=run.read(WORK/'configs/ablations/ue_staged_uav_experiments_20261028.json')
    protocol=run.read(WORK/matrix['protocol_file'])
    run.dump(work/'matrix.json',matrix);run.dump(work/'protocol.json',protocol);run.dump(root/'orders/manifest.json',{})
    audited=run.audit_matrix(matrix)
    m=dict(root=str(root),runtime_source=str(source),source_sha256={},protocol_path='protocol.json',
        protocol_sha256=run.canonical_sha(protocol),protocol=protocol,matrix_path='matrix.json',
        matrix_sha256=run.sha(work/'matrix.json'),recipes=audited['recipes'],base_lock=None,
        order_manifest_sha256=run.sha(root/'orders/manifest.json'),cache_receipts={})
    run.dump(root/'launch_manifest.json',m)
    monkeypatch.setattr(run,'REPO',source);monkeypatch.setattr(run,'WORK',work)
    monkeypatch.setattr(run.controller,'validate',Mock())
    with pytest.raises(RuntimeError,match='Missing or changed cache receipt'):
        run.validate_manifest(root,'C')
