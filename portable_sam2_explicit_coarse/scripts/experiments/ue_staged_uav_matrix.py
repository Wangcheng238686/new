"""Pure experiment resolver/auditor: no torch, data I/O, queue or training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[2]
MATRIX=WORK/'configs/ablations/ue_staged_uav_experiments_20261028.json'
SCIENCE=('arm','uav_view_ids','rgb_content','geometry_content')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_matrix(path=MATRIX):
    return json.loads(Path(path).read_text())


def resolve_recipes(matrix,base_arm=None):
    if set(matrix['recipes'])!=set('SUTC'):
        raise ValueError('Exactly four structure recipes required')
    if set(matrix['followup_changes'])!={'mod_constant','mod_no_geometry','views_1','views_3'}:
        raise ValueError('Unexpected follow-up conditions')
    allowed={'name','arm','uav_view_ids','rgb_content','geometry_content','subset_cache_root'}
    if any(set(r)!=allowed for r in matrix['recipes'].values()):
        raise ValueError('Recipe contains missing or undeclared configuration fields')
    for name,change in matrix['followup_changes'].items():
        expected={'rgb_content'} if name=='mod_constant' else {'geometry_content'} if name=='mod_no_geometry' else {'uav_view_ids'}
        if set(change)!=expected:raise ValueError('Hidden scientific changes in '+name)
    recipes=deepcopy(matrix['recipes'])
    if base_arm is not None:
        if base_arm not in ('U','T','C'):raise ValueError('Follow-up base must be U/T/C, never S')
        for name,change in matrix['followup_changes'].items():
            recipe=deepcopy(recipes[base_arm]);recipe.update(change);recipe['name']=name
            if name.startswith('views_'):recipe['subset_cache_root']=matrix['cache_root']
            recipes[name]=recipe
    for name,r in recipes.items():
        if r['name']!=name:raise ValueError('Recipe name mismatch')
        r.pop('recipe_sha256',None);r['recipe_sha256']=canonical_sha(r)
    return recipes


def delta(a,b):
    return {key:dict(reference=a[key],intervention=b[key]) for key in SCIENCE if a[key]!=b[key]}


def audit_matrix(matrix,base_arm=None):
    expected={'structure':['S','U','T','C'],'modality':['mod_constant','mod_no_geometry'],'views':['views_1','views_3']}
    if matrix['stages']!=expected:raise ValueError('Unexpected stage members')
    recipes=resolve_recipes(matrix,base_arm)
    for arm in 'SUTC':
        r=recipes[arm]
        if r['arm']!=arm or r['uav_view_ids']!=[0,72,144,216,288] or r['rgb_content']!='correct' or r['geometry_content']!='on' or r['subset_cache_root'] is not None:
            raise ValueError('Structure stage contains hidden modality/view/cache changes')
    pairs=[]
    # Arm expands to two explicit factors, rather than pretending U is one
    # operator. S->C changes BOTH; it is a system comparison, not one factor.
    factors=dict(S=(False,False),U=(True,False),T=(False,True),C=(True,True))
    for a,b,key in [('S','U','upstream_package'),('S','T','tail_evidence'),('U','C','tail_evidence'),('T','C','upstream_package')]:
        va,vb=factors[a],factors[b]
        if sum(x!=y for x,y in zip(va,vb))!=1:raise ValueError('Invalid structure edge')
        pairs.append(dict(reference=a,intervention=b,changed_factor=key,
            scope='U includes interaction, instance loss and proposal supplementation' if key=='upstream_package' else 'UBR evidence port only'))
    if base_arm:
        base=recipes[base_arm]
        for name,key in [('mod_constant','rgb_content'),('mod_no_geometry','geometry_content'),('views_1','uav_view_ids'),('views_3','uav_view_ids')]:
            changed=delta(base,recipes[name])
            if set(changed)!={key}:raise ValueError('Hidden scientific changes in '+name+': '+str(changed))
            r=recipes[name]
            if key=='uav_view_ids':
                expected_ids=[0] if name=='views_1' else [0,144,288]
                if r[key]!=expected_ids or r['subset_cache_root']!=matrix['cache_root']:raise ValueError('Wrong subset contract')
            elif r['subset_cache_root']!=base['subset_cache_root']:raise ValueError('Modality accidentally changes cache')
            if name=='mod_constant' and r[key]!='constant':raise ValueError('Wrong RGB intervention')
            if name=='mod_no_geometry' and r[key]!='off':raise ValueError('Wrong geometry intervention')
            pairs.append(dict(reference=base_arm,intervention=name,changed_factor=key,delta=changed))
    else:
        for arm in ('U','T','C'):audit_matrix(matrix,arm)
    return dict(passed=True,base_arm=base_arm,recipes=recipes,comparisons=pairs,
                deferred_stages=[] if base_arm else ['modality','views'],
                caution='U is a system package; geometry-off retains depth correspondence/support; view count is one fixed subset per count')
