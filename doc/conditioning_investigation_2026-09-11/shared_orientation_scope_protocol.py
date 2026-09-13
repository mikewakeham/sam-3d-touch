"""Pure-Python protocol for the matched generator-scope continuation."""
import re
from visual_dropout_protocol import dropout_schedule


def additional_shape_parameter(name):
    """Backbone-relative names; never include visual encoders or layout branches."""
    if re.fullmatch(r'(t_embedder|d_embedder)\.mlp\.(0|2)\.(weight|bias)', name):
        return True
    if re.fullmatch(r'latent_mapping\.shape\.(input_layer|out_layer)\.(weight|bias)', name):
        return True
    if re.fullmatch(r'blocks\.\d+\.adaLN_modulation\.1\.(weight|bias)', name):
        return True
    if re.fullmatch(r'blocks\.\d+\.self_attn\.(to_qkv|to_out|q_rms_norm|k_rms_norm)\.shape\..+', name):
        return True
    if re.fullmatch(r'blocks\.\d+\.mlp\.shape\..+', name):
        return True
    return False


def continuation_schedule():
    # A fresh balanced schedule; both arms get exactly the same 1000 updates.
    return dropout_schedule(30)


def next_noise_seed(local_step):
    if not 1 <= local_step <= 1000:
        raise ValueError('Expected continuation step 1..1000')
    return 29 + 1000 + local_step
