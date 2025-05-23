
def set_two_domain_input(images, inputs, domain, device):
    if (domain is None) or (domain == 'both'):
        images.real_a = inputs[0].to(device, non_blocking = True)
        images.real_b = inputs[1].to(device, non_blocking = True)

    # no [0] or [1] because we need the batch dim
    elif domain in [ 'a', 0 ]:
        #images.real_a = inputs[0].to(device, non_blocking = True)
        images.real_a = inputs.to(device, non_blocking = True)

    elif domain in [ 'b', 1 ]:
        #images.real_b = inputs[1].to(device, non_blocking = True)
        images.real_b = inputs.to(device, non_blocking = True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a' (alias 0), 'b' (alias 1), or 'both'"
        )

def set_two_domain_label(labels, inputs, domain, device):
    if (domain is None) or (domain == 'both'):
        labels.real_a = inputs[0].to(device, non_blocking = True)
        labels.real_b = inputs[1].to(device, non_blocking = True)

    # no [0] or [1] because we need the batch dim
    elif domain in [ 'a', 0 ]:
        labels.real_a = inputs.to(device, non_blocking = True)

    elif domain in [ 'b', 1 ]:
        labels.real_b = inputs.to(device, non_blocking = True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a' (alias 0), 'b' (alias 1), or 'both'"
        )