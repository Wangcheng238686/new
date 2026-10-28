"""Small CPU fixtures only: no COCO matching or459-image bootstraps."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
PATH=Path(__file__).resolve().parents[1]/'scripts/analysis/ue_staged_uav_analysis_20261028.py'
spec=importlib.util.spec_from_file_location('staged_analysis_test',PATH);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)


def manifest(root='/study',base=None):
    recipes={a:dict(arm=a,uav_view_ids=[0,72,144,216,288],rgb_content='correct',geometry_content='on',recipe_sha256=a) for a in 'SUTC'}
    if base:
        for name in ('mod_constant','mod_no_geometry','views_1','views_3'):recipes[name]=dict(recipes[base],recipe_sha256=name)
    return dict(root=str(root),recipes=recipes,base_lock={'base_arm':base} if base else None,
        stages=dict(structure=list('SUTC'),modality=['mod_constant','mod_no_geometry'],views=['views_1','views_3']))


def test_structure_edges_and_same_draw_interaction():
    plan=api.resolve_stage(manifest(),'structure')
    assert plan['names']==list('SUTC')
    assert set(plan['contrasts'])=={'U-S','T-S','C-S','C-U','C-T','C-U-T+S'}
    weights=plan['contrasts']['C-U-T+S'];assert weights==dict(C=1,U=-1,T=-1,S=1)
    points={k:{'AP':v} for k,v in dict(S=.4,U=.5,T=.6,C=.75).items()}
    # Common identical image fluctuations cancel exactly; independent draws
    # would introduce false variance in this fixture.
    fluctuations=np.linspace(-.03,.03,api.DRAWS)
    boot={k:fluctuations+v['AP'] for k,v in points.items()}
    out=api.contrast_result(points,boot,weights)
    assert out['delta']==pytest.approx(.05)
    assert out['paired_image_bootstrap95']==pytest.approx([.05,.05])
    assert out['bootstrap_samples']==500 and out['seed']==20261021


@pytest.mark.parametrize('base',['U','T','C'])
def test_followup_contributions_are_complete_minus_deprived(base):
    m=manifest(base=base)
    modality=api.resolve_stage(m,'modality');views=api.resolve_stage(m,'views')
    assert modality['names']==[base,'mod_constant','mod_no_geometry']
    assert modality['contrasts'][base+'-mod_constant']=={base:1,'mod_constant':-1}
    assert views['contrasts'][base+'-views_1']=={base:1,'views_1':-1}
    assert views['contrasts'][base+'-views_3']=={base:1,'views_3':-1}


def test_unresolved_or_wrong_base_fails():
    with pytest.raises(ValueError,match='Locked'):api.resolve_stage(manifest(),'modality')
    m=manifest(base='C');m['recipes']['C']['rgb_content']='constant'
    with pytest.raises(ValueError,match='complete correct'):api.resolve_stage(m,'views')
    m=manifest();m['stages']['structure']=['S','U','T']
    with pytest.raises(ValueError):api.resolve_stage(m,'structure')


def test_paired_identity_GT_and_unique_order():
    a=[dict(image_id=i,gt={'boxes':[i]},dt={}) for i in range(2)]
    b=json.loads(json.dumps(a));api.validate_records({'a':a,'b':b},images=2)
    with pytest.raises(ValueError,match='order'):api.validate_records({'a':a,'b':b[::-1]},images=2)
    b[1]['gt']['boxes']=[9]
    with pytest.raises(ValueError,match='GT differs'):api.validate_records({'a':a,'b':b},images=2)
    b[1]['image_id']=0
    with pytest.raises(ValueError,match='unique'):api.validate_records({'a':a,'b':b},images=2)


def test_plan_only_no_evaluation_needed_and_complete_refusal(tmp_path,capsys):
    m=manifest(tmp_path);(tmp_path/'launch_manifest.json').write_text(json.dumps(m))
    api.run(tmp_path,'structure',True)
    result=json.loads(capsys.readouterr().out)
    assert result['plan_only'] and len(result['metrics'])==4 and result['seed']==20261021
    assert not (tmp_path/'analysis').exists()
    folder=tmp_path/'analysis/structure';folder.mkdir(parents=True);(folder/'summary.json').write_text('{"complete":true}')
    with pytest.raises(RuntimeError,match='Preserve'):api.run(tmp_path,'structure')


def test_receipt_checkpoint_records_binding_and_bad_marker_rejection(tmp_path):
    m=manifest(tmp_path);plan=api.resolve_stage(m,'structure')
    (tmp_path/'launch_manifest.json').write_text(json.dumps(m))
    rows=[dict(image_id=i,gt={'bboxes':[]},dt={}) for i in range(459)]
    for name in plan['names']:
        folder=tmp_path/name;folder.mkdir();checkpoint=folder/'best_model.pth';checkpoint.write_bytes(name.encode())
        digest=api.sha(checkpoint);(folder/'training_complete.json').write_text(json.dumps(dict(best_checkpoint_sha256=digest,best={'epoch':73})))
        out=tmp_path/'evaluation/test'/name;out.mkdir(parents=True)
        (out/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=digest)))
        (out/'records.json').write_text(json.dumps(rows))
    controller=SimpleNamespace(validate=Mock(),accepted=Mock(return_value=True))
    data,evidence,checkpoints=api.load_inputs(tmp_path,m,plan,controller)
    assert set(data)==set('SUTC') and len(evidence)==17
    assert all(checkpoints[n]['sha256']==api.sha(tmp_path/n/'best_model.pth') for n in 'SUTC')
    controller.validate.assert_called_once_with(m)
    assert controller.accepted.call_count==8
    controller.accepted.return_value=False
    with pytest.raises(RuntimeError,match='Accepted'):api.load_inputs(tmp_path,m,plan,controller)
    controller.accepted.return_value=True
    (tmp_path/'C/best_model.pth').write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='Checkpoint receipt mismatch'):api.load_inputs(tmp_path,m,plan,controller)
