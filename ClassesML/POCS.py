import torch


def pocs_postprocess(x, hyperparameters):
    """
    POCS (Projection Onto Convex Sets) post-processing, applied after the
    AutoEncoder reconstruction to enforce known physical constraints on
    the spectrogram (see course IP26_9).

    3 constraints, applied as alternating projections for n_iter rounds:
        C1 - energy >= 0                          (clip negative values)
        C2 - energy only in [f_min_bin, f_max_bin] (zero out band mask)
        C3 - no energy outside the clip            (zero first/last frame)

    Args:
        x : Tensor (B, 1, F, T)
        hyperparameters : dict, keys pocs_n_iter / pocs_f_min_bin / pocs_f_max_bin

    Returns:
        Tensor (B, 1, F, T), same device as x.
    """
    n_iter    = hyperparameters.get("pocs_n_iter",    10)
    f_min_bin = hyperparameters.get("pocs_f_min_bin",  2)
    f_max_bin = hyperparameters.get("pocs_f_max_bin", 50)

    s = x.clone()

    for _ in range(n_iter):
        s = _project_c1_nonnegative(s)
        s = _project_c2_bandlimited(s, f_min_bin, f_max_bin)
        s = _project_c3_finite_support(s)

    return s


def _project_c1_nonnegative(s):
    return torch.clamp(s, min=0.0)


def _project_c2_bandlimited(s, f_min_bin, f_max_bin):
    mask = torch.zeros_like(s)
    mask[:, :, f_min_bin:f_max_bin, :] = 1.0
    return s * mask


def _project_c3_finite_support(s):
    s = s.clone()
    s[:, :, :,  0] = 0.0  
    s[:, :, :, -1] = 0.0 
    return s
