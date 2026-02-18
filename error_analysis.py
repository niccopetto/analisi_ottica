"""
Error Analysis Module for Optical Wavefront Reconstruction.

Provides analytical error propagation for Zernike coefficients, RMS, and Strehl ratio.
All uncertainties are configurable via the ERROR_CONFIG dictionary.

Usage:
    from error_analysis import compute_error_budget
    budget = compute_error_budget(zernike_results, phases_for_zernike,
                                  matrix_Z_list, phi_vec_list, wavelength)
"""

import numpy as np
from numpy import pi

# Unit helpers (same as gs_engine)
mm = 1e-3
nm = 1e-9

# =============================================================================
# DEFAULT ERROR CONFIGURATION — Edit these values for your setup
# =============================================================================
ERROR_CONFIG = {
    # Optical bench parameters
    'f1':             25.0 * mm,       # Short lens focal length
    'delta_f1':       2.5 * mm,        # ±10% on f1
    'f2':             2000.0 * mm,     # Long lens focal length
    'delta_f2':       100.0 * mm,      # ±5% on f2
    'f_oap':          1000.0 * mm,     # OAP focal length
    'delta_f_oap':    10.0 * mm,       # ±1% on f_oap
    'w_oap':          90.0 * mm,       # Beam waist on OAP
    'delta_w_oap':    10.0 * mm,       # ±1cm uncertainty
    'delta_angle_oap': 1.0,            # ±1° orientation error (degrees)

    # Measurement uncertainties
    'delta_distance': 5.0 * mm,        # ±5mm on propagation distances
    'pixel_size':     0.0084 * mm,     # Effective pixel size
    'delta_pixel':    0.5e-3 * mm,     # ±0.5µm on pixel size
    'roi_shift_px':   3,               # ±3 pixels ROI selection uncertainty
}


def _lstsq_covariance(matrix_Z, phi_vec, coeffs, n_terms):
    """
    Compute Zernike coefficient uncertainties from least-squares covariance.

    The covariance matrix of the lstsq solution is:
        cov(c) = σ² × (ZᵀZ)⁻¹
    where σ² = ||residuals||² / (N - n_terms)
    """
    residuals = phi_vec - matrix_Z @ coeffs
    N = len(phi_vec)
    dof = max(N - n_terms, 1)
    sigma2 = np.sum(residuals**2) / dof

    try:
        ZtZ_inv = np.linalg.inv(matrix_Z.T @ matrix_Z)
        cov = sigma2 * ZtZ_inv
        delta_coeffs = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        delta_coeffs = np.full(n_terms, np.nan)

    return delta_coeffs, sigma2


def _distance_sensitivity(n_terms, wavelength, config):
    """
    Estimate how ±δd in propagation distance affects Zernike coefficients.

    Defocus (Z4) is most sensitive: a distance error δd introduces
    a quadratic phase error equivalent to a defocus change.
    In the Fresnel approximation:
        δφ_defocus ≈ π × δd / (λ × N²_fresnel)

    Higher-order terms are less affected; we use empirical decay factors.
    """
    delta_d = config['delta_distance']
    px = config['pixel_size']

    # Fresnel number approximation for typical image size (512 pixels)
    typical_N = 512
    a = typical_N * px  # aperture half-size
    N_fresnel = a**2 / (wavelength * 1.0)  # at ~1m distance

    # Defocus sensitivity (dominant term)
    delta_Z4 = pi * delta_d / (wavelength * max(N_fresnel, 1.0))

    # Build sensitivity vector: defocus-dominated, with decay for higher orders
    delta_coeffs = np.zeros(n_terms)
    if n_terms > 3:
        delta_coeffs[3] = delta_Z4  # Z4: Defocus
    # Higher orders decay roughly as 1/n² relative to defocus
    for j in range(4, n_terms):
        n_radial = int(np.ceil(np.sqrt(j)))
        delta_coeffs[j] = delta_Z4 / max(n_radial**2, 1)

    return delta_coeffs


def _pixel_sensitivity(coeffs, config):
    """
    Pixel size uncertainty propagation.

    Phase scales as px² (through propagation kernel), so:
        δc_j / c_j ≈ 2 × δpx / px
    """
    px = config['pixel_size']
    delta_px = config['delta_pixel']

    relative_error = 2.0 * delta_px / px
    delta_coeffs = np.abs(coeffs) * relative_error

    return delta_coeffs


def _bg_noise_propagation(matrix_Z, sigma_bg, n_terms):
    """
    Background noise propagation through lstsq.

    If the phase has additive noise with std = σ_bg, the coefficient
    uncertainty is: δc_bg = σ_bg × √(diag((ZᵀZ)⁻¹))
    """
    try:
        ZtZ_inv = np.linalg.inv(matrix_Z.T @ matrix_Z)
        delta_coeffs = sigma_bg * np.sqrt(np.abs(np.diag(ZtZ_inv)))
    except np.linalg.LinAlgError:
        delta_coeffs = np.full(n_terms, np.nan)

    return delta_coeffs


def _roi_shift_sensitivity(n_terms, R_pupil, config):
    """
    ROI selection uncertainty: a shift of ±roi_shift_px in the center
    position introduces spurious tilt and coma.

    Tilt sensitivity: δZ₂ ≈ δZ₃ ≈ 2π × δr / R_pupil
    Coma sensitivity: δZ₇ ≈ δZ₈ ≈ π × (δr / R_pupil)²
    """
    shift = config['roi_shift_px']

    delta_coeffs = np.zeros(n_terms)

    tilt_err = 2 * pi * shift / max(R_pupil, 1.0)
    if n_terms > 1:
        delta_coeffs[1] = tilt_err   # Z2: Tilt X
    if n_terms > 2:
        delta_coeffs[2] = tilt_err   # Z3: Tilt Y

    coma_err = pi * (shift / max(R_pupil, 1.0))**2
    if n_terms > 6:
        delta_coeffs[6] = coma_err   # Z7: Coma X
    if n_terms > 7:
        delta_coeffs[7] = coma_err   # Z8: Coma Y

    return delta_coeffs


def _oap_angle_sensitivity(n_terms, config):
    """
    OAP angle misalignment introduces primarily coma and astigmatism.

    For a parabolic mirror tilted by δθ:
        Coma ≈ (δθ / 4F#) in waves, where F# = f / (2×w_oap)
        Astigmatism ≈ (δθ² / 4F#²) in waves (second order, usually small)
    """
    delta_theta = config['delta_angle_oap'] * pi / 180  # deg → rad
    f_oap = config['f_oap']
    w_oap = config['w_oap']

    F_number = f_oap / (2 * w_oap)

    delta_coeffs = np.zeros(n_terms)

    # Coma from tilt (in radians of phase)
    coma_rad = 2 * pi * delta_theta / (4 * F_number)
    if n_terms > 6:
        delta_coeffs[6] = coma_rad   # Z7: Coma X
    if n_terms > 7:
        delta_coeffs[7] = coma_rad   # Z8: Coma Y

    # Astigmatism (second-order, smaller)
    astig_rad = 2 * pi * delta_theta**2 / (4 * F_number**2)
    if n_terms > 4:
        delta_coeffs[4] = astig_rad  # Z5: Astigmatism
    if n_terms > 5:
        delta_coeffs[5] = astig_rad  # Z6: Astigmatism

    return delta_coeffs


def expected_waist_at_focus(wavelength, config):
    """
    Gaussian beam waist at OAP focus and its uncertainty.

    Model: w_focus = λ × f_OAP / (π × w_OAP)

    Error propagation:
        (δw/w)² = (δf/f)² + (δw_oap/w_oap)²
    """
    f_oap = config['f_oap']
    w_oap = config['w_oap']
    delta_f = config['delta_f_oap']
    delta_w = config['delta_w_oap']

    w_focus = wavelength * f_oap / (pi * w_oap)

    # Relative error (quadrature)
    rel_err = np.sqrt((delta_f / f_oap)**2 + (delta_w / w_oap)**2)
    delta_w_focus = w_focus * rel_err

    return w_focus, delta_w_focus


def compute_error_budget(zernike_results, phases_for_zernike,
                         matrix_Z_list, phi_vec_list,
                         wavelength, config=None):
    """
    Compute analytical error budget for Zernike coefficients, RMS, and Strehl.

    Parameters
    ----------
    zernike_results : list of dict
        Each dict has 'Name', 'Coeffs', 'Radius_Used', 'Valid_Mask'.
    phases_for_zernike : list of dict
        Each dict has 'Phase_Data', 'Radius_px', 'Local_CX', 'Local_CY'.
    matrix_Z_list : list of np.ndarray
        Zernike design matrices from the decomposition (one per plane).
    phi_vec_list : list of np.ndarray
        Phase data vectors from the decomposition (one per plane).
    wavelength : float
        Wavelength in meters.
    config : dict, optional
        Error configuration. Uses ERROR_CONFIG defaults if None.

    Returns
    -------
    dict with keys:
        'delta_coeffs'     : (n_planes, n_terms) total coefficient errors [rad]
        'delta_rms'        : (n_planes,) RMS errors [waves]
        'delta_strehl'     : (n_planes,) Strehl ratio errors
        'expected_waist'   : float, expected waist at focus [m]
        'delta_waist'      : float, waist uncertainty [m]
        'budget_breakdown' : dict of per-source contributions
    """
    if config is None:
        config = ERROR_CONFIG

    n_planes = len(zernike_results)
    if n_planes == 0:
        return {
            'delta_coeffs': np.array([]),
            'delta_rms': np.array([]),
            'delta_strehl': np.array([]),
            'expected_waist': 0, 'delta_waist': 0,
            'budget_breakdown': {}
        }

    n_terms = len(zernike_results[0]['Coeffs'])

    # Storage for all planes
    all_delta_coeffs = np.zeros((n_planes, n_terms))
    all_delta_rms = np.zeros(n_planes)
    all_delta_strehl = np.zeros(n_planes)

    # Per-source breakdown storage
    breakdown = {
        'fit':      np.zeros((n_planes, n_terms)),
        'distance': np.zeros((n_planes, n_terms)),
        'pixel':    np.zeros((n_planes, n_terms)),
        'bg':       np.zeros((n_planes, n_terms)),
        'roi':      np.zeros((n_planes, n_terms)),
        'oap':      np.zeros((n_planes, n_terms)),
    }

    # OAP angle contribution (same for all planes)
    delta_oap = _oap_angle_sensitivity(n_terms, config)

    # Distance contribution (same structure for all planes)
    delta_dist = _distance_sensitivity(n_terms, wavelength, config)

    for i, (res, item) in enumerate(zip(zernike_results, phases_for_zernike)):
        coeffs = res['Coeffs']
        R_pupil = res['Radius_Used']
        matrix_Z = matrix_Z_list[i]
        phi_vec = phi_vec_list[i]

        # --- 1. lstsq fit covariance ---
        delta_fit, sigma2 = _lstsq_covariance(matrix_Z, phi_vec, coeffs, n_terms)
        breakdown['fit'][i] = delta_fit

        # --- 2. Distance sensitivity ---
        breakdown['distance'][i] = delta_dist

        # --- 3. Pixel size sensitivity ---
        delta_px = _pixel_sensitivity(coeffs, config)
        breakdown['pixel'][i] = delta_px

        # --- 4. BG noise propagation ---
        # Estimate σ_bg from the phase residuals (noise floor)
        sigma_bg = np.sqrt(sigma2) if sigma2 > 0 else 0
        delta_bg = _bg_noise_propagation(matrix_Z, sigma_bg, n_terms)
        breakdown['bg'][i] = delta_bg

        # --- 5. ROI shift sensitivity ---
        delta_roi = _roi_shift_sensitivity(n_terms, R_pupil, config)
        breakdown['roi'][i] = delta_roi

        # --- 6. OAP angle ---
        breakdown['oap'][i] = delta_oap

        # --- Total: quadrature sum ---
        delta_total = np.sqrt(
            delta_fit**2 + delta_dist**2 + delta_px**2 +
            delta_bg**2 + delta_roi**2 + delta_oap**2
        )
        all_delta_coeffs[i] = delta_total

        # --- RMS error propagation ---
        # RMS² = Σ c_j² (j ≥ 3, excluding piston and tilt)
        # δRMS = (1/RMS) × √(Σ (c_j × δc_j)²)
        c_aberr = coeffs[3:]  # from Z4 onwards (skip piston, tilt_x, tilt_y)
        dc_aberr = delta_total[3:]

        rms_rad = np.sqrt(np.sum(c_aberr**2))
        if rms_rad > 0:
            delta_rms_rad = np.sqrt(np.sum((c_aberr * dc_aberr)**2)) / rms_rad
        else:
            delta_rms_rad = 0

        # Convert to waves
        rms_waves = rms_rad / (2 * pi)
        delta_rms_waves = delta_rms_rad / (2 * pi)
        all_delta_rms[i] = delta_rms_waves

        # --- Strehl error propagation ---
        # Strehl = exp(−(2π × RMS_waves)²) = exp(−RMS_rad²)
        # δStrehl = Strehl × 2 × RMS_rad × δRMS_rad
        strehl = np.exp(-(rms_rad**2))
        delta_strehl = strehl * 2 * rms_rad * delta_rms_rad
        all_delta_strehl[i] = delta_strehl

    # --- Expected waist at focus ---
    w_focus, delta_w_focus = expected_waist_at_focus(wavelength, config)

    return {
        'delta_coeffs': all_delta_coeffs,       # (n_planes, n_terms) in rad
        'delta_rms': all_delta_rms,              # (n_planes,) in waves
        'delta_strehl': all_delta_strehl,        # (n_planes,)
        'expected_waist': w_focus,               # meters
        'delta_waist': delta_w_focus,            # meters
        'budget_breakdown': breakdown,           # per-source dict
    }
