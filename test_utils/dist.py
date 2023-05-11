from torch.distributed import _broadcast_coalesced
from torch.distributed.distributed_c10d import _get_default_group

# used for intra-node param sync and inter-node sync as well
broadcast_bucket_size = int(250 * 1024 * 1024)

def sync_params_and_buffers(module, src_rank=0):
    module_states = []

    if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
        parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
    else:
        parameters_to_ignore = []

    for name, param in module.named_parameters():
        if name not in parameters_to_ignore:
            module_states.append(param.detach())

    for name, buffer in module.named_buffers():
        if name not in parameters_to_ignore:
            module_states.append(buffer.detach())

    if len(module_states) > 0:
        _broadcast_coalesced(_get_default_group(), module_states, broadcast_bucket_size, src_rank)
