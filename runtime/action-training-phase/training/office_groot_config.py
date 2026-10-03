"""Office MuJoCo SO101 embodiment: inputs/outputs in radians, gripper included.

NVIDIA processor converts five arm targets relative to CURRENT measured state,
then decodes back to absolute radians. These are not consecutive FLUX deltas.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig,ActionFormat,ActionRepresentation,ActionType,ModalityConfig

office_config={
    'video':ModalityConfig(delta_indices=[0],modality_keys=['scene','wrist']),
    'state':ModalityConfig(delta_indices=[0],modality_keys=['single_arm','gripper']),
    'action':ModalityConfig(delta_indices=list(range(16)),modality_keys=['single_arm','gripper'],action_configs=[
        ActionConfig(rep=ActionRepresentation.RELATIVE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT)]),
    'language':ModalityConfig(delta_indices=[0],modality_keys=['annotation.human.task_description'])}
register_modality_config(office_config,embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
