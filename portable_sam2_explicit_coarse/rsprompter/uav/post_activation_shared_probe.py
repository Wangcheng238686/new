"""N2 with an explicit shared enhanced feature for mask logits and UDPR."""
import weakref
from .post_activation_probe import PostActivationAdapter


class SharedPostActivationAdapter(PostActivationAdapter):
    def __init__(self,pretrained,direction='N2'):
        super().__init__(pretrained,direction)
        self.sync_capture = True

    def attach(self,model):
        self._wrapper_ref = weakref.ref(model.roi_head.mask_head.mask_decoder)
        super().attach(model)

    def _inject(self,module,args,base):
        enhanced = super()._inject(module,args,base)
        if self.sync_capture:
            # The wrapper's original capture hook has already fired. Both the
            # decoder hypernetwork and UDPR must receive this exact tensor.
            self._wrapper_ref()._captured_upscaled_embedding = enhanced
        return enhanced
