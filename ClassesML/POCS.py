import torch


def pocs_postprocess(x, hyperparameters):
    """
    Apply Projection Onto Convex Sets (POCS) post-processing to a batch of spectrograms.

    Application to audio classification:
        After reconstruction by the AutoEncoder, the output spectrogram may violate
        known physical constraints of bird vocalisations. POCS restores consistency
        by projecting the reconstructed spectrogram onto three convex sets:

            C1 — Bounded signals :
                Energy values must be non-negative (log-compressed power cannot be
                physically negative after reconstruction). Projection: value clipping.

            C2 — Band-limited signals :
                Malaysian garden birds vocalise primarily between ~500 Hz and ~8 kHz.
                Energy outside this band is noise. Projection: band-pass binary mask
                on the frequency axis (generalisation of the low-pass filter case).

            C3 — Finite-support signals :
                The clip is exactly 3 seconds; no acoustic energy should exist outside
                the clip boundaries. Projection: zero-masking of boundary frames.

    Args:
        x               : Tensor of shape (B, 1, F, T) — batch of spectrograms.
        hyperparameters : dict with optional keys:
                            "pocs_n_iter"    — number of alternating projection cycles (default: 10).
                            "pocs_f_min_bin" — lower frequency bin index of the pass band (default: 2).
                            "pocs_f_max_bin" — upper frequency bin index of the pass band (default: 50).

    Returns:
        Tensor of shape (B, 1, F, T) on the same device as the input.

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


# -----------------------------------------------------------------------
# Individual projection operators
# -----------------------------------------------------------------------

def _project_c1_nonnegative(s):
    """
    Project onto C1: the set of signals bounded below by zero.
    """
    return torch.clamp(s, min=0.0)


def _project_c2_bandlimited(s, f_min_bin, f_max_bin):
    """
    Project onto C2: the set of band-limited signals within [f_min_bin, f_max_bin].
    """
    mask = torch.zeros_like(s)
    mask[:, :, f_min_bin:f_max_bin, :] = 1.0
    return s * mask


def _project_c3_finite_support(s):
    """
    Project onto C3: the set of finite-support signals.
    """
    s = s.clone()
    s[:, :, :,  0] = 0.0   # zero first frame
    s[:, :, :, -1] = 0.0   # zero last frame
    return s
