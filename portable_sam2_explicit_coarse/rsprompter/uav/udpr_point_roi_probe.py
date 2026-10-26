"""UAV evidence enters only the existing UDPR's selected point features."""
import torch
from torch import nn
from torch.nn import functional as F
from .local_adapter_variants import UnitLocalUAVAdapter


def mask_indices_to_satellite_uv(indices, height, width, boxes):
    """Native roi_local output cell centers -> image edge coordinates."""
    local = torch.stack(((indices % width + .5) / width,
                         (indices // width + .5) / height), -1)
    return boxes[:,None,:2] + local * (boxes[:,None,2:]-boxes[:,None,:2])


class UDPRPointROIAdapter(UnitLocalUAVAdapter):
    def __init__(self, pretrained, direction='S1'):
        if direction != 'S1': raise ValueError(direction)
        super().__init__(pretrained)
        self.direction = direction
        self.query.requires_grad_(False)  # Satellite image-Q is not used here.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026102301)
            self.point_query = nn.Linear(32,32)
            self.output = nn.Linear(32,32)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        # Zero output projection gives exact hotstart, without a second trainable
        # scalar shutting its gradient off. Alpha is retained for replay gates.
        self.alpha.data.fill_(1.)
        self.alpha.requires_grad_(False)
        self.observe = False
        self.observation = {}
        self.point_calls = 0
        self.audit_gt = None
        self.point_diagnostics = []

    def encode(self, rgb, index, geom, anchor, satellite_high):
        super().encode(rgb,index,geom,satellite_high)
        self.stats = {}; self.observation = {}
        self.point_diagnostics = []

    def attach(self, model):
        assert not self.handles, 'Duplicate point adapter attachment'
        assert model.roi_head.mask_head.final_mask_coordinate_mode == 'roi_local'
        tail = model.roi_head.mask_head.decoder_tail_refiner
        assert tail is not None and tail.num_points == 64
        self.tail = None  # Do not register the frozen satellite module as a child.
        def hook(module,args): return self._point_hook(module,args)
        self.handles.append(tail.register_forward_pre_hook(hook))
        self.handles.append(model.roi_head.mask_head.register_forward_pre_hook(self._boxes_hook,with_kwargs=True))
        self.handles.append(tail.register_forward_hook(self._diagnostic_hook))

    def _diagnostic_hook(self, tail, args, output):
        if not self.enabled or self.context is None or not self.observe or self.audit_gt is None:return
        with torch.no_grad():
            # Direct forward bypasses hooks: original frozen UDPR on the same
            # raw logits/tokens and original satellite features, no recursion.
            _,base = tail.forward(args[0],self.original_high,args[2])
            actual = output[1]
            assert torch.equal(base['indices'],actual['indices'])
            ids = actual['indices'];h,w = args[0].shape[-2:]
            uv = mask_indices_to_satellite_uv(ids,h,w,self.boxes)
            boxes,masks = self.audit_gt
            boxes = boxes.to(self.boxes.device)
            a=self.boxes[:,None];b=boxes[None]
            inter=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
            iou=inter/((a[...,2:]-a[...,:2]).prod(-1)+(b[...,2:]-b[...,:2]).prod(-1)-inter).clamp_min(1e-6)
            if not boxes.numel():return
            best,match=iou.max(-1);valid=best>=.5
            masks=masks.to(ids.device)
            y=masks[match[:,None],uv[...,1].long().clamp(0,511),uv[...,0].long().clamp(0,511)].bool()
            old=base['refined_selected_logits']>=tail.boundary_logit
            new=actual['refined_selected_logits']>=tail.boundary_logit
            supported=self.selected_support & valid[:,None]
            included=valid[:,None].expand_as(y)
            self.point_diagnostics.append(dict(matched_rois=int(valid.sum()),rois=len(valid),
                matched_points=int(included.sum()),supported_matched_points=int(supported.sum()),
                corrected=int(((old!=y)&(new==y)&included).sum()),
                damaged=int(((old==y)&(new!=y)&included).sum()),
                changed=int(((old!=new)&included).sum()),
                old_correct=int(((old==y)&included).sum()),new_correct=int(((new==y)&included).sum())))
            totals={k:sum(x[k] for x in self.point_diagnostics) for k in self.point_diagnostics[0]}
            self.observation.update(point_diagnostics=totals,
                point_diagnostic_contract='same raw mask and selected points; original UDPR vs UAV-conditioned UDPR; best GT box IoU>=.5; repeated RoIs counted; native-grid GT nearest; deployment threshold')

    def _point_hook(self, tail, args):
        if not self.enabled or self.context is None:return
        logits, high, tokens = args
        f,index,geom,_ = self.context
        n,c,h,w = high.shape
        assert c == 32 and logits.shape == (n,1,h,w)
        selected,_ = tail._select(logits)
        if self.observe:self.original_high=high
        uv = mask_indices_to_satellite_uv(selected,h,w,self.boxes)
        cell = (uv[...,1]//4).long().clamp(0,127)*128+(uv[...,0]//4).long().clamp(0,127)
        ids = index[cell]; valid = (ids >= 0) & ((uv>=0)&(uv<512)).all(-1)[...,None]
        self.selected_support=valid.any(-1)
        selected_sat = high.flatten(2).transpose(1,2).gather(1,selected[...,None].expand(-1,-1,c))
        q = self.point_query(selected_sat)
        safe = ids.clamp_min(0)
        k = self.key(f)[safe]+self.geom_key(geom[cell])
        v = self.value(f)[safe]+self.geom_value(geom[cell])
        score = (F.normalize(q.float(),dim=-1)[...,None,:]*F.normalize(k.float(),dim=-1)).sum(-1)*4
        weights = score.masked_fill(~valid,-1e4).softmax(-1)*valid
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled = (weights[...,None]*v).sum(-2)
        delta = self.output(F.layer_norm(pooled.float(),(32,)).to(q.dtype))*valid.any(-1)[...,None]
        applied = (self.alpha*delta).to(high.dtype)
        updated = high.flatten(2).scatter_add(2,selected[:,None,:].expand(-1,c,-1),applied.transpose(1,2)).reshape_as(high)
        self.point_calls += 1
        with torch.no_grad():
            self.stats = dict(point_support_fraction=float(valid.any(-1).float().mean()),
                source_delta_rms=float(delta.float().square().mean().sqrt()),
                applied_delta_rms=float(applied.float().square().mean().sqrt()),
                base_point_rms=float(selected_sat.float().square().mean().sqrt()),
                rois=n,points_per_roi=selected.shape[1])
            if self.observe:
                self.observation = dict(scope='latest UDPR call; all selected points, including unsupported',
                    supported_points=int(valid.any(-1).sum()),points=selected.numel(),
                    coordinate_contract='roi_local mask cell centers mapped through actual proposal bbox',
                    sparse_changes_only=True)
        # Raw logits and tokens are exactly the original objects; selection and
        # bbox/detection are not modified by this branch.
        return logits,updated,tokens
