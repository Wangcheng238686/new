"""Same normalized evidence and state as L0; only the residual receiver moves."""
from .attention_ladder import AttentionLadderAdapter


class PostActivationAdapter(AttentionLadderAdapter):
    def __init__(self, pretrained, direction='L0'):
        if direction not in ('L0', 'N2'):
            raise ValueError(direction)
        super().__init__(pretrained, direction='L0')
        self.injection_site = direction
        self.injection_calls = 0

    def attach(self, model):
        assert not self.handles, 'Do not attach this adapter twice'
        if self.injection_site == 'L0':
            return super().attach(model)
        head = model.roi_head.mask_head
        self.handles.append(head.register_forward_pre_hook(self._boxes_hook, with_kwargs=True))
        self.handles.append(head.mask_decoder.decoder.output_upscaling[-1].register_forward_hook(self._inject))

    def _inject(self, module, args, base):
        self.injection_calls += 1
        return super()._inject(module, args, base)
