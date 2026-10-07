"""Pinned same-pool OCR convergence: LR is the only training intervention."""
from itertools import islice
from .frozen_ocr_study import bucket_schedule,SHA
from .ocr_pool_expansion import state_digest

POOL_SHA='d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9'
PARENT_HEAD='checkpoints/iam_ocr_pool_expansion/20261007-041949/expanded8192/head-best.pt'
PARENT_SHA='5de8792405583c7651f83de12ac7300f398fa5b088c5d0570ad00bb9f22b75ed'
PARENT_UPDATES=12000


def validate_parent(saved,pool):
    c=saved['config'];splits=pool['splits']
    if saved['updates']!=PARENT_UPDATES or c['pool_manifest_sha256']!=POOL_SHA:raise ValueError('pinned8192 step12000 parent required')
    if c['source_sha256']!=SHA or c['feature_mode']!='relative_scaled' or c['attention_radius'] is not None:raise ValueError('codec/OCR feature contract changed')
    for field,key in [('train_ids','large_train'),('dev_ids','dev'),('held_out_ids','held_out'),('feature_calibration_ids','small_train')]:
        if c[field]!=splits[key]:raise ValueError('fixed pool split/calibration changed: '+field)
    probe=splits['small_train'][::6]
    if c['common_train_probe']!=probe or c['posterior_evaluation_ids']!=probe+splits['dev']+splits['held_out']:raise ValueError('evaluation probe/noise population changed')
    if c['schedule_seed']!=43 or c['parent_updates']!=6000 or c['physical_batch']!=16:raise ValueError('parent data iterator contract changed')
    if any(g['lr']!=1e-4 for g in saved['optimizer_state_dict']['param_groups']):raise ValueError('inherited LR1e-4 required')
    return c


def resumed_schedule(cache,ids,steps,skip=6000,seed=43,batch_size=16):
    """Regenerate the exact parent iterator then resume AFTER its consumed prefix."""
    if not isinstance(skip,int) or skip<0 or not isinstance(steps,int) or steps<1:raise ValueError('nonnegative iterator skip and positive steps required')
    return islice(bucket_schedule(cache,ids,skip+steps,batch_size=batch_size,seed=seed),skip,None)


def set_lr_only(optimizer,lr):
    """Change group LR without resetting any moment, counter or other setting."""
    import copy,math
    if not isinstance(lr,(int,float)) or not math.isfinite(lr) or lr<=0:raise ValueError('finite positive LR required')
    before=copy.deepcopy(optimizer.state_dict())
    for g in optimizer.param_groups:g['lr']=float(lr)
    after=copy.deepcopy(optimizer.state_dict())
    # Ignore ONLY lr when comparing the complete serialized optimizer state.
    for state in (before,after):
        for group in state['param_groups']:group.pop('lr')
    if state_digest(before)!=state_digest(after):raise AssertionError('LR intervention changed other optimizer state')
    return dict(lr=float(lr),optimizer_except_lr_sha256=state_digest(after),all_moments_counters_other_settings_preserved=True)
