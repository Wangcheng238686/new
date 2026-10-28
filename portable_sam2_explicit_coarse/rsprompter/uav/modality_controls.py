"""Complete-path content controls, preserving measured geometric correspondence.

These are appearance/geometry-CONTENT ablations, not removal of measured depth:
indices, continuous anchors, support, view identity and spatial offsets remain.
The controls own no model parameters, losses or runtime decisions.
"""
from dataclasses import dataclass
import torch

RGB_MEAN=(123.675,116.28,103.53)


@dataclass(frozen=True)
class ModalityControls:
    rgb_content: str='correct'
    geometry_content: str='on'

    def __post_init__(self):
        if self.rgb_content not in ('correct','constant'):raise ValueError('rgb_content must be correct or constant')
        if self.geometry_content not in ('on','off'):raise ValueError('geometry_content must be on or off')

    def validate_arm(self,arm):
        if arm not in ('S','U','T','C'):raise ValueError('Expected S/U/T/C')
        if arm=='S' and (self.rgb_content!='correct' or self.geometry_content!='on'):
            raise ValueError('S has no UAV inputs and cannot use modality controls')

    def apply_payload(self,payload):
        """No mutation of cached inputs. Default content preserves tensor identity.

        Only manipulated fields are validated here; common input loaders and
        model entry points own the unchanged payload's existing validations.
        """
        result=dict(payload)
        if self.rgb_content=='constant':
            rgb=payload['rgb']
            if not torch.is_tensor(rgb) or rgb.ndim!=4 or rgb.shape[1]!=3 or not rgb.is_floating_point():
                raise ValueError('Constant RGB requires floating [V,3,H,W] raw pixel values')
            # Raw RGB ImageNet means, not zero pixels: downstream normalization
            # produces exactly zero content. All canonical slots are replaced.
            mean=rgb.new_tensor(RGB_MEAN).view(1,3,1,1)
            result['rgb']=mean.expand_as(rgb).clone()
        if self.geometry_content=='off':
            geometry=payload['geometry']
            if not torch.is_tensor(geometry) or geometry.ndim!=4 or geometry.shape[1]!=5:
                raise ValueError('Geometry content control requires [B,5,H,W]')
            geometry=geometry.clone();geometry[:,:4]=0
            result['geometry']=geometry
        return result

    def apply_candidate_geometry(self,geometry):
        if self.geometry_content=='on':return geometry
        if not torch.is_tensor(geometry) or geometry.ndim<1 or geometry.shape[-1]!=7:
            raise ValueError('Candidate geometry must end in seven descriptor channels')
        result=geometry.clone();result[..., [2,3,6]]=0
        return result

    def audit(self):
        return dict(rgb_content=self.rgb_content,geometry_content=self.geometry_content,
            rgb_constant_raw_mean=list(RGB_MEAN) if self.rgb_content=='constant' else None,
            geometry5_zero_channels=[0,1,2,3] if self.geometry_content=='off' else [],
            candidate7_zero_channels=[2,3,6] if self.geometry_content=='off' else [],
            geometry5_preserved_channels=[4] if self.geometry_content=='off' else list(range(5)),
            candidate7_preserved_channels=[0,1,4,5] if self.geometry_content=='off' else list(range(7)),
            measured_index_anchor_preserved=True,support_preserved=True,
            scope='all upstream, auxiliary/candidate, native-tail and formal-inference paths',
            interpretation='geometry-content control; not depth-free: measured correspondence, support, offsets and yaw retained',
            extra_parameters=0,extra_losses=0)
