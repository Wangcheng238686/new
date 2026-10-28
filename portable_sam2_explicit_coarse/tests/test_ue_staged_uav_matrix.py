"""Explicit scientific deltas, shared full100 budget and no hidden factorial arms."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
WORK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matrix_under_test',WORK/'scripts/experiments/ue_staged_uav_matrix.py')
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)


@pytest.mark.parametrize('base',['U','T','C'])
def test_all_followups_change_only_one_declared_factor(base):
    source=matrix.load_matrix();audit=matrix.audit_matrix(source,base)
    assert audit['passed'] and len(audit['recipes'])==8
    recipes=audit['recipes']
    for name,key in [('mod_constant','rgb_content'),('mod_no_geometry','geometry_content'),('views_1','uav_view_ids'),('views_3','uav_view_ids')]:
        assert set(matrix.delta(recipes[base],recipes[name]))=={key}
    assert recipes['mod_constant']['subset_cache_root'] is None
    assert recipes['mod_no_geometry']['subset_cache_root'] is None
    assert recipes['views_1']['subset_cache_root']==source['cache_root']


def test_hidden_structure_modality_or_sampling_changes_rejected():
    original=matrix.load_matrix()
    for key,value in [('rgb_content','constant'),('geometry_content','off'),('uav_view_ids',[0]),('subset_cache_root','wrong')]:
        altered=deepcopy(original);altered['recipes']['U'][key]=value
        with pytest.raises(ValueError):matrix.audit_matrix(altered)
    with pytest.raises(ValueError):matrix.resolve_recipes(original,'S')


def test_coupled_modality_intervention_rejected():
    altered=matrix.load_matrix();altered['followup_changes']['mod_constant']['geometry_content']='off'
    with pytest.raises(ValueError,match='Hidden scientific'):matrix.audit_matrix(altered,'C')


def test_budget_same_as_existing_full100_and_no_new_losses():
    old=json.loads((WORK/'configs/ablations/ue_g_ubr_native_full_protocol_20261028.json').read_text())
    new=json.loads((WORK/matrix.load_matrix()['protocol_file']).read_text())
    for key in ('epochs','world_size','per_rank_batch','per_rank_accumulation','effective_batch','used_images_per_epoch','updates_per_epoch','total_updates','seed','lr','weight_decay','warmup_updates','cosine_schedule_epochs','cosine_min_factor','clip','amp_init_scale','AMP_training','EMA','max_dets','RPN_pre','augmentation','sample_order','native_mask','UBR_loss','instance_head','validation_epochs','checkpoint_epochs','selection','evaluation','resume','native_precision'):
        assert old[key]==new[key],key


def test_undeclared_recipe_fields_or_extra_experiments_rejected():
    altered=matrix.load_matrix();altered['recipes']['C']['learning_rate']=123
    with pytest.raises(ValueError,match='undeclared'):matrix.audit_matrix(altered)
    altered=matrix.load_matrix();altered['recipes']['extra']=dict(altered['recipes']['C'])
    with pytest.raises(ValueError,match='Exactly four'):matrix.audit_matrix(altered)
