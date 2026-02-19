
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import re
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from numpy import pi
import poppy.zernike as zernike  # used for zernike.zernike(n, m, rho, theta)

# Custom modules
import AuxiliaryTools as at
from gs_engine import GerSaxPhaseRetriever, mm, nm
# Toggle error analysis on/off (set to True to enable error bars)
ENABLE_ERROR_ANALYSIS = False

if ENABLE_ERROR_ANALYSIS:
    try:
        from error_analysis import compute_error_budget
        HAS_ERROR_ANALYSIS = True
    except ImportError:
        HAS_ERROR_ANALYSIS = False
        print("[INFO] error_analysis.py not found, error budget disabled.")
else:
    HAS_ERROR_ANALYSIS = False
import tkinter as tk
from tkinter import filedialog

def estrai_distanza_da_obj(img_obj):
    # img_obj.imageName contains the filename
    match = re.search(r'(\d+)', os.path.basename(img_obj.imageName))
    return int(match.group(1)) if match else 0

def super_gaussian_curve(coords, A, x0, y0, wx, wy, theta, P, offset):
    """
    Elliptical super-Gaussian intensity profile:
        I(x,y) = A * exp(-2 * ((x'/wx)^2 + (y'/wy)^2)^(P/2)) + offset
    where:
        x' =  (x-x0)*cos(θ) + (y-y0)*sin(θ)    (rotated coordinates)
        y' = -(x-x0)*sin(θ) + (y-y0)*cos(θ)
        wx, wy = beam waists (1/e² radii) along the principal axes
        θ = rotation angle of the ellipse (radians)
        P = super-Gaussian order (P=2 → standard Gaussian)
    """
    y, x = coords
    dx = x - x0
    dy = y - y0
    # Rotate into ellipse frame
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xp = dx * cos_t + dy * sin_t
    yp = -dx * sin_t + dy * cos_t
    # Elliptical radial coordinate
    r_ell_sq = (xp / wx)**2 + (yp / wy)**2
    g = A * np.exp(-2 * r_ell_sq**(P / 2)) + offset
    return g.ravel()

def noll_to_nm(j):
    """
    Converts the Noll sequential index (j) into the (n, m) pair.
    Supports up to the first 15 terms (or more if logic is expanded).
    """
    # Noll Index: (n, m)
    mapping = {
        1: (0, 0),   # Piston
        2: (1, 1),   # Tilt X
        3: (1, -1),  # Tilt Y
        4: (2, 0),   # Defocus
        5: (2, -2),  # Astigmatism 1st
        6: (2, 2),   # Astigmatism 2nd
        7: (3, -1),  # Coma X
        8: (3, 1),   # Coma Y
        9: (3, -3),  # Trefoil X
        10: (3, 3),  # Trefoil Y
        11: (4, 0),  # Spherical
        12: (4, 2),  # Secondary Astig
        13: (4, -2), # Secondary Astig
        14: (4, 4),  # Quadrafoil
        15: (4, -4)  # Quadrafoil
    }
    
    if j in mapping:
        return mapping[j]
    else:
        # Fallback for higher orders if needed, or raise error
        # Simple analytic approximation or expanded map required for j > 15
        raise ValueError(f"Index j={j} not mapped (extend the dictionary or use full algorithm)")

def get_circular_roi(image, cx, cy, radius):
    """
    1. Crops a square around the center.
    2. Applies a circular mask (NaN outside the radius).
    Returns: (cropped_masked_image, peak_to_valley, local_cx, local_cy)
    """
    h, w = image.shape
    
    # 1. Square limit definition, prevent out of bounds
    x1 = int(np.floor(max(0, cx - radius)))
    x2 = int(np.ceil(min(w, cx + radius)))
    y1 = int(np.floor(max(0, cy - radius)))
    y2 = int(np.ceil(min(h, cy + radius)))
    
    cropped_square = image[y1:y2, x1:x2].copy()
    
    # 2. Create circular mask
    h_c, w_c = cropped_square.shape
    Y, X = np.ogrid[:h_c, :w_c]
    
    # Local center coordinates within the cropped image
    local_cx = cx - x1
    local_cy = cy - y1
    
    dist_sq = (X - local_cx)**2 + (Y - local_cy)**2
    cropped_square = np.ma.masked_where(dist_sq > radius**2, cropped_square)
    
    # Calculate Peak to Valley
    if cropped_square.count() == 0:
        ptv = 0
    else:
        ptv = cropped_square.max() - cropped_square.min()

    return cropped_square, ptv, local_cx, local_cy

def preprocess(input_dir=".", pixelsizex=0.0084*mm, pixelsizey=0.0084*mm):
    """
    Performs image preprocessing: loading, interactive selection, resampling, ROI/BG subtraction.
    """
    print(f"Processing images from: {input_dir}")
    
    # Setup AuxiliaryClass
    ac = at.AuxiliaryClass()
    ac.setImagesDir(input_dir)
    
    # Find image files (.tif / .tiff, case-insensitive)
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(input_dir, ext)))
    all_files = sorted(list(set(all_files)))
    imagesFNs = [os.path.basename(f) for f in all_files]
    
    if not imagesFNs:
        print("No .tiff files found in directory!")
        return [], [], [], pixelsizex, pixelsizey

    ac.setImagesFilenames(*imagesFNs)
    
    # Input for resampling
    res_input = input("Enter the resampling factor (e.g., 0.5 for half size, default 0.5): ")
    resamplingFactor = float(res_input) if res_input.strip() else 0.5

    resampledpixelsizex = pixelsizex / resamplingFactor
    resampledpixelsizey = pixelsizey / resamplingFactor
    
    # Load images
    inputImages = [at.ImageClass() for _ in ac.imagesPath]
    for i, img in enumerate(inputImages):
        img.loadImage(ac.imagesPath[i], {'x': pixelsizex, 'y': pixelsizey})
    
    # --- Interactive channel selection ---
    channel_input = input("Select a channel among 'red','green','blue' (default: red): ").strip()
    if channel_input:
        channel = channel_input.capitalize()[0]
    else:
        channel = 'R'
    
    for i in range(len(inputImages)):
        inputImages[i].selectChannel(channel)
    
    # Show loaded images (before resampling)
    for img in inputImages:
        img.showOriginal()
        
    # Resample and show results
    print(f"\nResampling with factor {resamplingFactor}...")
    for img in inputImages:
        img.resampleOriginal(factor=resamplingFactor)
        img.showOriginal()
        print(f"  {img.imageName} max value: {img.imageData.max()}")
        
    # ROI and BG selection
    print("\n--- ROI and Background Selection ---")
    
    for img in inputImages:
        print(f"Selecting for {img.imageName}...")
        while True:
            img.selectROI()
            img.showROI()
            img.selectBG()
            img.retrieveAverageBG()
            redo = input(f"Do you want to redo the ROI selection for {img.imageName}? (y/n, default n): ")
            if redo.strip().lower() != 'y':
                break
        img.subtractBGfromROI()
    
    # --- Optional ROI resampling ---
    ROIresampling = input("Would you like to resample back the ROI? (y/n, default n): ").strip().capitalize()
    if ROIresampling and ROIresampling[0] == 'Y':
        resROIfactor = 1 / resamplingFactor
        for img in inputImages:
            img.resampleROI(factor=resROIfactor)
    
    # --- BG verification after subtraction ---
    print("\n--- Background Verification ---")
    for img in inputImages:
        winName = f"Check {img.imageName} ROI background"
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        ROIedges = cv2.selectROI(winName, img.ROI, True, False)
        tmp = img.ROI[ROIedges[1]:ROIedges[1]+ROIedges[3], ROIedges[0]:ROIedges[0]+ROIedges[2]]
        tmp_avr_bg = np.mean(tmp)
        print(f"  {img.imageName} BG value after correction: {tmp_avr_bg}")
        cv2.destroyWindow(winName)
        
    # --- Final image dimensions (separate x and y) ---
    npix_x_input = input("Enter the number of pixels in x direction for the final image (default 1024): ")
    finalImageNpixelsx = int(npix_x_input) if npix_x_input.strip() else 1024
    npix_y_input = input("Enter the number of pixels in y direction for the final image (default 1024): ")
    finalImageNpixelsy = int(npix_y_input) if npix_y_input.strip() else 1024

    # --- Extent / coordinate calculation ---
    xMin = resampledpixelsizex / 2.0
    xMax = resampledpixelsizex * (finalImageNpixelsx - 0.5)
    yMin = resampledpixelsizey / 2.0
    yMax = resampledpixelsizey * (finalImageNpixelsy - 0.5)

    xCoord = np.linspace(xMin, xMax, finalImageNpixelsx)
    yCoord = np.linspace(yMin, yMax, finalImageNpixelsy)

    xExtent = resampledpixelsizex / 2.0 + np.max(xCoord)
    yExtent = resampledpixelsizey / 2.0 + np.max(yCoord)

    print(f"Final image actual size: {xExtent:f}, {yExtent:f}")

    # --- Center of mass and build final images ---
    for img in inputImages:
        img.getCenterOfMassOfROI()
        img.buildImageWithSize(finalImageNpixelsx, finalImageNpixelsy)
        img.showCenteredROI()
        
    # Ordering and Distance Calculation
    print("\n--- Calculating Distances ---")
    valori_grezzi = np.array([estrai_distanza_da_obj(img) for img in inputImages])
    ordine_calcolato = np.argsort(valori_grezzi)
    valori_ordinati = valori_grezzi[ordine_calcolato]
    
    delta_raw = np.abs(np.diff(valori_ordinati))
    
    # Magnification Factor for Z axis 
    # (Standard: 1.0. If lens system is used, Z_eff = Z_raw * M_longitudinal)
    msg = "Enter Z-axis Magnification Factor (default 1.0): "
    z_mag_input = input(msg)
    z_mag = float(z_mag_input) if z_mag_input.strip() else 1.0
    
    dist_scale = 0.1 * z_mag # 0.1 mm step * Magnification
    d = (delta_raw * dist_scale) * mm 
    
    print(f"\n--- Magnification applied: {z_mag} ---\n") 
    

    # Cleaning
    for i, img in enumerate(inputImages):
        img.centeredROI = np.maximum(img.centeredROI, 0)
        
    return inputImages, d, ordine_calcolato, resampledpixelsizex, resampledpixelsizey

def main():
    # Configuration
    InputDIR = "./imgs_folder/te_22_05_cannone/"
    wavelength = 633 * nm
    # Initial pixelsize
    pixelsizex = 0.0084 * mm
    pixelsizey = 0.0084 * mm
    
    print("Starting Optical Analysis...")
    
    # 1. Preprocessing
    # Force folder selection "a priori"
    print("Opening folder selection dialog...")
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Force window to be on top
    InputDIR = filedialog.askdirectory(title="Select Input Directory")
    root.destroy()
    
    if not InputDIR:
        print("No directory selected. Exiting.")
        return 
    
    # Call preprocess (User Interaction required for ROI)
    inputImages, d, ordine_calcolato, res_px, res_py = preprocess(InputDIR, pixelsizex, pixelsizey)
    
    if not inputImages:
        print("No images to process.")
        return
    
    # --- Density map plot of preprocessed images ---
    Nimages = len(inputImages)
    fig, ax = plt.subplots(1, Nimages, figsize=(12, 4*Nimages))
    if Nimages == 1:
        ax = [ax]
    for i in range(Nimages):
        Ny, Nx = inputImages[i].ysize, inputImages[i].xsize
        yMax_plot = (Ny + .5) * inputImages[i].roiCalibration['y']
        xMax_plot = (Nx + .5) * inputImages[i].roiCalibration['x']
        img_plot = ax[i].imshow(inputImages[i].centeredROI,
                  extent=np.array([0, xMax_plot, 0, yMax_plot]) * 1e3,
                  cmap='RdBu_r',
                  aspect='equal',
                  origin='lower',
                  vmin=0,
                  vmax=inputImages[i].centeredROI.max())
        ax[i].set_xlabel('x [mm]')
        ax[i].set_ylabel('y [mm]')
        ax[i].set_title(inputImages[i].imageName)
        plt.colorbar(img_plot, shrink=.3)
        print(inputImages[i].imageName,
              'shape: ' + str(inputImages[i].centeredROI.shape),
              'calibration: ' + str(inputImages[i].roiCalibration),
              f"xMax = {xMax_plot:.2f}, yMax = {yMax_plot:.2f}",
              sep="\n\t-",
              end="\n")
    fig.tight_layout()
    plt.show()

    # --- Propagation Report ---
    print("\n--- SEQUENCE REPORT ---")
    for i, idx in enumerate(ordine_calcolato):
        nome_f = os.path.basename(inputImages[idx].imageName)
        if i < len(d):
            step_m = d[i]
            print(f"[{i+1}] {nome_f} \n    |  ↓  Propagation: {step_m*1000:.1f} mm ({step_m:.4f} m)")
        else:
            print(f"[{i+1}] {nome_f} (Final Plane / Far Field)")
    
    # --- Image sanitization ---
    print("\n--- IMAGE CLEANING ---")
    for i, img in enumerate(inputImages):
        min_val = np.min(img.centeredROI)
        print(f"Img {i}: Original minimum = {min_val:.4f}")
        img.centeredROI = np.maximum(img.centeredROI, 0)
        print(f"Img {i}: Post-cleaning minimum = {np.min(img.centeredROI):.4f}")
    print("\nImages sanitized. NaNs should be gone now.")

    # 2. GS Algorithm
    print("\nStarting GS algorithm...")
    GS = GerSaxPhaseRetriever(*inputImages,
                              distances=d,
                              wavelength=wavelength, 
                              ordering=ordine_calcolato)
    
    GS.GS_algorithm(Niterations=200)
    
    # --- GS Diagnostics ---
    campo = GS.FarField.field
    ci_sono_nan = np.isnan(campo).any()
    ci_sono_inf = np.isinf(campo).any()
    max_val = np.max(np.abs(campo))
    print(f"--- DIAGNOSTICS ---")
    print(f"Are there NaNs? {ci_sono_nan}")
    print(f"Are there Infs? {ci_sono_inf}")
    print(f"Max Intensity: {max_val:.4e}")
    print(f"-------------------")
    
    # --- Plot Retrieved Phases + Intensities (Mosaic) ---
    n_imgs = len(GS.imagesNames)
    ax_titles = [f"retrieved phase: {name}" for name in GS.imagesNames]
    ax_titles_copy = [f"retrieved intensity: {name}" for name in GS.imagesNames]
    fig, ax = plt.subplot_mosaic([ax_titles, ax_titles_copy], figsize=(16, 8))
    extent = np.array([0, GS.size, 0, GS.size]) * 1e3
    for i, name in enumerate(GS.imagesNames):
        ax[ax_titles_copy[i]].imshow(GS.RetrieveIntensity(name), cmap='afmhot', vmin=0, extent=extent, origin='lower')
        is_img = ax[ax_titles[i]].imshow(GS.RetrievePhase(name, unwrap=True), cmap='seismic', 
                                          vmax=2*pi, vmin=-2*pi, extent=extent, origin='lower')
        fig.colorbar(is_img, shrink=.5, location='bottom')
        ax[ax_titles[i]].set_title(ax_titles[i])
        ax[ax_titles[i]].set_xlabel('x [mm]')
        ax[ax_titles[i]].set_ylabel('y [mm]')
    fig.tight_layout()
    plt.show()

    # --- Plot GS Convergence (Relative %) ---
    fig_conv, ax_conv = plt.subplots(1, 1)
    ax_conv.plot(GS.MSError[1:] / GS.MSError[1] * 100, 'o--', ms=4, label='Mean Square Error')
    ax_conv.plot(GS.CEError[1:] / GS.CEError[1] * 100, 'o--', ms=4, label='Relative Entropy')
    ax_conv.set_ylabel('Error(%)')
    ax_conv.legend()
    plt.show()
    
    # =====================================================================
    # 3. SUPER-GAUSSIAN FIT
    # =====================================================================
    print("\n--- SUPER-GAUSSIAN FIT ---")
    
    # Data prep: Normalize RAW images and create grids
    normalized_input_intensity = []
    gs_retrieved_intensity = []
    Y = []
    X = []
    for i, name in enumerate(GS.imagesNames):
        # Raw camera data (for fitting)
        idx = ordine_calcolato[i]
        I_raw = inputImages[idx].centeredROI.astype(float)
        I_raw = I_raw / np.max(I_raw)
        normalized_input_intensity.append(I_raw)
        # GS Retrieved (for visual comparison only)
        I_gs = GS.RetrieveIntensity(name)
        I_gs = I_gs / np.max(I_gs)
        gs_retrieved_intensity.append(I_gs)
        h_i, w_i = I_raw.shape
        temp_y, temp_x = np.mgrid[0:h_i, 0:w_i]
        Y.append(temp_y)
        X.append(temp_x)
    print(f"Data ready. Loaded {len(normalized_input_intensity)} matrices (raw camera)")

    # Intensity fitting
    fit_intensity = []
    for I_norm, grid_y, grid_x, name in zip(normalized_input_intensity, Y, X, GS.imagesNames):
        # Smoothing (used only for dynamic waist guess)
        I_smooth = gaussian_filter(I_norm, sigma=3)
        
        # Dynamic waist guess
        area_pixel = np.sum(I_norm > 0.10)
        w_guess = np.sqrt(area_pixel / np.pi)
        w_guess = max(w_guess, 3.0)
        
        # Max location from raw data for center guess
        y_max, x_max = np.unravel_index(np.argmax(I_norm), I_norm.shape)
        
        # Theta guess from intensity-weighted covariance (image moments)
        weights = I_norm / I_norm.sum()
        cov_xx = np.sum((grid_x - x_max)**2 * weights)
        cov_yy = np.sum((grid_y - y_max)**2 * weights)
        cov_xy = np.sum((grid_x - x_max) * (grid_y - y_max) * weights)
        theta_guess = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
        p0 = [1.0, x_max, y_max, w_guess, w_guess, theta_guess, 2.0, 0.0]
        bounds = ([0.5, 0, 0, 1.0, 1.0, -np.pi, 1.0, 0], 
                  [1.5, grid_x.shape[1], grid_x.shape[0], 
                   grid_x.shape[1]/2, grid_x.shape[0]/2, np.pi, 20.0, 0.5])
        try:
            popt, pcov = curve_fit(super_gaussian_curve, (grid_y, grid_x), I_norm.ravel(), p0=p0, bounds=bounds)
            A, x0, y0, wx_fit, wy_fit, theta_fit, P_fit, offset = popt
            # Normalize: ensure wx >= wy (swap if needed, adjust angle)
            if wy_fit > wx_fit:
                wx_fit, wy_fit = wy_fit, wx_fit
                theta_fit += np.pi / 2
            # Keep angle in [-π/2, π/2]
            theta_fit = ((theta_fit + np.pi/2) % np.pi) - np.pi/2
            theta_deg = np.degrees(theta_fit)
            print(f"{name:<20} | wx={wx_fit:.2f}  wy={wy_fit:.2f}  θ={theta_deg:+.1f}°  | P={P_fit:.4f}  | ({x0:.1f}, {y0:.1f})")
            fit_intensity.append({
                'Name': name,
                'Waist_X': wx_fit,
                'Waist_Y': wy_fit,
                'Waist': max(wx_fit, wy_fit),  # backward compat: largest waist
                'Theta': theta_fit,
                'P': P_fit,
                'Center_X': x0,
                'Center_Y': y0,
                'Amplitude': A,
                'Offset': offset
            })
        except RuntimeError:
            print(f"{name:<20} | FIT FAILED")
    print("-" * 70)

    # --- Intensity Fit Mosaic (Raw / GS Retrieved / Fit / Residuals) ---
    n_fit = len(normalized_input_intensity)
    row_raw  = [f"raw_{i}"  for i in range(n_fit)]
    row_meas = [f"meas_{i}" for i in range(n_fit)]
    row_fit  = [f"fit_{i}"  for i in range(n_fit)]
    row_res  = [f"res_{i}"  for i in range(n_fit)]
    layout = [row_raw, row_meas, row_fit, row_res]
    fig_fit, axd = plt.subplot_mosaic(layout, figsize=(20, 13), constrained_layout=True)
    print("Generating plot with Mosaic...")

    for i, (I_raw_norm, params, gy, gx) in enumerate(zip(normalized_input_intensity, fit_intensity, Y, X)):
        # Model reconstruction
        args = (params['Amplitude'], params['Center_X'], params['Center_Y'], 
                params['Waist_X'], params['Waist_Y'], params['Theta'],
                params['P'], params['Offset'])
        I_model = super_gaussian_curve((gy, gx), *args).reshape(gy.shape)
        
        # Get GS retrieved image for comparison
        I_gs = gs_retrieved_intensity[i]
        
        # Residual (Fit vs Raw)
        residuals = I_raw_norm - I_model
        limit = np.max(np.abs(residuals))
        
        # Pixel error
        pixel_error = np.mean(np.abs(residuals)) * 100
        params['Pixel_Error'] = pixel_error

        # Row 0: Raw Input (camera data)
        key_raw = f"raw_{i}"
        axd[key_raw].imshow(I_raw_norm, cmap='inferno', vmin=0, vmax=1)
        axd[key_raw].set_title(f"Raw: {params['Name']}", fontsize=10, color='green')

        # Row 1: GS Retrieved (what algorithm predicts)
        key_m = f"meas_{i}"
        im_m = axd[key_m].imshow(I_gs, cmap='inferno', vmin=0, vmax=1)
        axd[key_m].set_title(f"GS Retrieved: {params['Name']}", fontsize=10)
        
        # Row 2: Elliptical Fit
        key_f = f"fit_{i}"
        im_f = axd[key_f].imshow(I_model, cmap='inferno', vmin=0, vmax=1)
        theta_deg = np.degrees(params['Theta'])
        axd[key_f].set_title(f"Fit (wx={params['Waist_X']:.1f}, wy={params['Waist_Y']:.1f}, θ={theta_deg:+.0f}°)", fontsize=9, color='blue')
        
        # Row 3: Residuals
        key_r = f"res_{i}"
        im_r = axd[key_r].imshow(residuals, cmap='bwr', vmin=-limit, vmax=limit)
        axd[key_r].set_title(f"Residui | Errore: {pixel_error:.2f}%", fontsize=9, fontweight='bold')
        
        for key in [key_raw, key_m, key_f, key_r]:
            axd[key].set_xticks([])
            axd[key].set_yticks([])

    axd["raw_0"].set_ylabel("Raw Input", fontsize=14, fontweight='bold')
    axd["meas_0"].set_ylabel("GS Retrieved", fontsize=14, fontweight='bold')
    axd["fit_0"].set_ylabel("Fit Model", fontsize=14, fontweight='bold')
    axd["res_0"].set_ylabel("Residuals", fontsize=14, fontweight='bold')
    cbar_ax1 = fig_fit.add_axes([1.01, 0.4, 0.015, 0.5])
    fig_fit.colorbar(im_m, cax=cbar_ax1, label='Norm. Intensity')
    cbar_ax2 = fig_fit.add_axes([1.01, 0.05, 0.015, 0.25])
    fig_fit.colorbar(im_r, cax=cbar_ax2, label='Residuals')
    plt.suptitle("Comparison: Measurement vs Super-Gaussian", fontsize=16)
    plt.show()

    # =====================================================================
    # 4. CROP AND PHASE MASKING
    # =====================================================================
    print("\n--- CROP AND PHASE MASKING ---")
    print("Starting ROI extraction on Phase...")
    
    phases_for_zernike = []
    
    n_fit = len(fit_intensity)
    row_raw = [f"raw_{i}" for i in range(n_fit)]
    row_crop = [f"crop_{i}" for i in range(n_fit)]
    fig_roi, axd_roi = plt.subplot_mosaic([row_raw, row_crop], figsize=(20, 8), constrained_layout=True)

    for i, params in enumerate(fit_intensity):
        name = params['Name']
        w_fit = params['Waist']
        cx = params['Center_X']
        cy = params['Center_Y']
        
        # ROI radius (2x waist)
        roi_radius = w_fit * 2.0
        
        # Retrieve unwrapped phase
        raw_phase = GS.RetrievePhase(name, unwrap=True)
        
        # Apply circular ROI (also returns the local center coordinates)
        masked_phase, ptv_value, local_cx, local_cy = get_circular_roi(raw_phase, cx, cy, roi_radius)
        
        phases_for_zernike.append({
            'Name': name,
            'Phase_Data': masked_phase,
            'PtV': ptv_value,
            'Radius_px': roi_radius,
            'Local_CX': local_cx,  # actual beam center within the crop
            'Local_CY': local_cy
        })
        
        # Plot: Raw phase
        axd_roi[f"raw_{i}"].imshow(raw_phase, cmap='jet')
        axd_roi[f"raw_{i}"].set_title(f"Full Phase: {name}\n(Unwrapped)", fontsize=9)
        circle = plt.Circle((cx, cy), roi_radius, color='white', fill=False, lw=2, ls='--')
        axd_roi[f"raw_{i}"].add_patch(circle)
        axd_roi[f"raw_{i}"].axis('off')

        # Plot: Cropped phase
        im_crop = axd_roi[f"crop_{i}"].imshow(masked_phase, cmap='jet')
        axd_roi[f"crop_{i}"].set_title(f"ROI (2w)\nPtV: {ptv_value:.2f} rad", fontsize=10, fontweight='bold')
        axd_roi[f"crop_{i}"].axis('off')

    axd_roi["raw_0"].set_ylabel("Total Phase", fontsize=14, fontweight='bold')
    axd_roi["crop_0"].set_ylabel("Crop & Mask", fontsize=14, fontweight='bold')
    plt.suptitle("ROI Extraction for Zernike Analysis (Radius = 2 * Waist)", fontsize=16)
    plt.show()

    # =====================================================================
    # 5. ZERNIKE DECOMPOSITION (Least Squares)
    # =====================================================================
    # NOTE: We use np.linalg.lstsq instead of poppy's decompose_opd.
    # decompose_opd computes coefficients via simple dot-product projection:
    #   c_j = sum(OPD * Z_j) / N_pixels
    # This assumes an ORTHONORMAL basis, which is only valid on a perfectly
    # centered, unobscured unit circle.
    # lstsq solves the overdetermined system ||Z*c - phi||^2 -> min,
    # which does NOT require orthonormality and is therefore more robust
    # for irregular/cropped/off-center apertures like our circular ROI.
    print("\n--- ZERNIKE DECOMPOSITION ---")
    
    n_terms = 15
    zernike_results = []
    matrix_Z_list = []   # saved for error analysis
    phi_vec_list = []    # saved for error analysis
    
    print(f"Starting Zernike decomposition on {len(phases_for_zernike)} planes...")
    
    for item in phases_for_zernike:
        name = item['Name']
        phase_data = item['Phase_Data']
        R_pupil = item['Radius_px']
        
        # Coordinate setup — use actual beam center from the fit, not h//2, w//2
        h_z, w_z = phase_data.shape
        y_z, x_z = np.indices((h_z, w_z))
        cx_z = item['Local_CX']  # real center from super-gaussian fit
        cy_z = item['Local_CY']
        
        r_pixel = np.sqrt((x_z - cx_z)**2 + (y_z - cy_z)**2)
        theta = np.arctan2(y_z - cy_z, x_z - cx_z)
        rho = r_pixel / R_pupil
        
        # Masking
        valid_mask = (~np.isnan(phase_data)) & (rho <= 1.0)
        phi_vec = phase_data[valid_mask]
        rho_vec = rho[valid_mask]
        theta_vec = theta[valid_mask]
        
        if len(phi_vec) == 0:
            print(f"Warning: No valid pixels for {name}")
            continue

        # Build the Zernike design matrix: each column is one Zernike polynomial
        # evaluated at the valid pixel coordinates (rho, theta)
        matrix_Z = []
        for j in range(1, n_terms + 1):
            n_z, m_z = noll_to_nm(j)
            Z_j = zernike.zernike(n_z, m_z, rho=rho_vec, theta=theta_vec)
            matrix_Z.append(Z_j)
        matrix_Z = np.array(matrix_Z).T  # shape: (N_valid_pixels, n_terms)
        
        # Solve for coefficients via least squares: minimize ||Z*c - phi||^2
        # This is equivalent to c = (Z^T Z)^{-1} Z^T phi (normal equations)
        coeffs, _, _, _ = np.linalg.lstsq(matrix_Z, phi_vec, rcond=None)
        
        zernike_results.append({
            'Name': name,
            'Coeffs': coeffs,
            'Radius_Used': R_pupil,
            'Valid_Mask': valid_mask  # cached for reconstruction
        })
        matrix_Z_list.append(matrix_Z)
        phi_vec_list.append(phi_vec)
        
        # Print coefficients in waves (λ) for optical convention
        defocus_waves = coeffs[3] / (2 * pi)
        astigmatism_waves = np.sqrt(coeffs[4]**2 + coeffs[5]**2) / (2 * pi)
        print(f"{name:<20} | Defocus (Z4): {defocus_waves:.4f} λ | Astig (Z5+Z6): {astigmatism_waves:.4f} λ")
    
    print("Calculation complete.")

    # =====================================================================
    # 6. ZERNIKE RECONSTRUCTION
    # =====================================================================
    print("\n--- ZERNIKE RECONSTRUCTION ---")
    
    n_zr = len(zernike_results)
    row_orig = [f"orig_{i}" for i in range(n_zr)]
    row_recon = [f"recon_{i}" for i in range(n_zr)]
    row_diff = [f"diff_{i}" for i in range(n_zr)]
    layout_recon = [row_orig, row_recon, row_diff]
    fig_recon, axd_recon = plt.subplot_mosaic(layout_recon, figsize=(20, 12), constrained_layout=True)
    
    print("Starting phase reconstruction from coefficients...")
    
    for i, (res, item) in enumerate(zip(zernike_results, phases_for_zernike)):
        name = res['Name']
        coeffs = res['Coeffs']
        R_pupil = res['Radius_Used']
        valid_mask = res['Valid_Mask']  # reuse cached mask
        phase_orig = item.get('Phase_Crop', item.get('Phase_Data'))
        
        # Reconstruct using actual beam center from the fit
        h_r, w_r = phase_orig.shape
        y_r, x_r = np.indices((h_r, w_r))
        cx_r = item['Local_CX']  # real center from super-gaussian fit
        cy_r = item['Local_CY']
        rho_r = np.sqrt((x_r - cx_r)**2 + (y_r - cy_r)**2) / R_pupil
        theta_r = np.arctan2(y_r - cy_r, x_r - cx_r)
        
        pupil_mask = (rho_r <= 1.0)
        phase_recon_flat = np.zeros(np.sum(pupil_mask))
        
        for j, c_j in enumerate(coeffs):
            n_z, m_z = noll_to_nm(j + 1)
            Z_j = zernike.zernike(n_z, m_z, rho=rho_r[pupil_mask], theta=theta_r[pupil_mask])
            phase_recon_flat += c_j * Z_j
        
        # Remapping
        phase_recon = np.full((h_r, w_r), np.nan)
        phase_recon[pupil_mask] = phase_recon_flat
        
        # Residual
        diff = phase_orig - phase_recon
        rms_error = np.sqrt(np.nanmean(diff**2))
        
        # Plot: Original
        im0 = axd_recon[f"orig_{i}"].imshow(phase_orig, cmap='jet')
        axd_recon[f"orig_{i}"].set_title(f"Original: {name}", fontsize=10)
        
        # Plot: Reconstructed
        im1 = axd_recon[f"recon_{i}"].imshow(phase_recon, cmap='jet')
        axd_recon[f"recon_{i}"].set_title(f"Reconstructed ({len(coeffs)} terms)", fontsize=10)
        
        # Plot: Residuals
        limit_r = np.nanmax(np.abs(diff)) if not np.all(np.isnan(diff)) else 1
        im2 = axd_recon[f"diff_{i}"].imshow(diff, cmap='bwr', vmin=-limit_r, vmax=limit_r)
        axd_recon[f"diff_{i}"].set_title(f"Residuals (RMS: {rms_error:.3f})", fontsize=9)
        
        for row in [f"orig_{i}", f"recon_{i}", f"diff_{i}"]:
            axd_recon[row].set_xticks([])
            axd_recon[row].set_yticks([])

    if n_zr > 0:
        axd_recon["orig_0"].set_ylabel("Original Phase", fontsize=12, fontweight='bold')
        axd_recon["recon_0"].set_ylabel("Reconstructed Phase", fontsize=12, fontweight='bold')
        axd_recon["diff_0"].set_ylabel("Difference", fontsize=12, fontweight='bold')
    plt.suptitle("Phase Comparison: Original vs Zernike Polynomial Reconstruction", fontsize=16)
    plt.show()

    # =====================================================================
    # 7. ERROR BUDGET (optional)
    # =====================================================================
    error_budget = None
    if HAS_ERROR_ANALYSIS:
        print("\n--- ERROR BUDGET ---")
        error_budget = compute_error_budget(
            zernike_results, phases_for_zernike,
            matrix_Z_list, phi_vec_list,
            wavelength
        )
        
        # Expected waist at focus vs fit waist
        w_exp = error_budget['expected_waist']
        dw_exp = error_budget['delta_waist']
        print(f"Expected waist at OAP focus: {w_exp*1e6:.1f} ± {dw_exp*1e6:.1f} µm")
        if fit_intensity:
            w_fit_px = np.mean([p['Waist'] for p in fit_intensity])
            w_fit_m = w_fit_px * pixelsizex
            print(f"Waist from fit (mean):     {w_fit_m*1e6:.1f} µm")
    else:
        print("\n[SKIP] Error budget not available.")

    # =====================================================================
    # 8. OPTICAL REPORT
    # =====================================================================
    print("\n--- OPTICAL REPORT ---")
    
    names = [res['Name'] for res in zernike_results]
    indices = np.arange(len(names))
    # Convert all Zernike coefficients from radians to waves (λ) for the report
    all_coeffs_rad = np.array([res['Coeffs'] for res in zernike_results])
    all_coeffs = all_coeffs_rad / (2 * pi)  # rad → waves (λ)
    
    rms_originale = []   # in waves
    rms_residua = []     # in waves
    strehl_ratio = []
    
    print("Starting metrics calculation (units: waves λ)...")
    
    for res, item in zip(zernike_results, phases_for_zernike):
        orig = item.get('Phase_Crop', item.get('Phase_Data'))
        if orig is None:
            print(f"Error: Phase data not found for {res['Name']}")
            continue
        coeffs_w = res['Coeffs'] / (2 * pi)  # rad → waves
        
        # RMS Originale (in waves)
        rms_o = np.nanstd(orig) / (2 * pi)
        rms_originale.append(rms_o)
        
        # RMS Residua (approximation, in waves)
        power_zernike = np.sum(coeffs_w[1:]**2) if len(coeffs_w) > 1 else 0
        estimated_variance = max(0, rms_o**2 - power_zernike)
        rms_r = np.sqrt(estimated_variance)
        rms_residua.append(rms_r)
        
        # Strehl Ratio (Maréchal approximation): σ must be in radians
        rms_r_rad = rms_r * (2 * pi)  # convert back to rad for Strehl
        strehl = np.exp(-(rms_r_rad**2))
        strehl_ratio.append(strehl)

    # Error bars (only if error budget is available)
    has_errbars = error_budget is not None
    if has_errbars:
        delta_coeffs_waves = error_budget['delta_coeffs'] / (2 * pi)
    
    # Plot 1: Aberration Evolution
    fig_evo, ax1_evo = plt.subplots(figsize=(12, 6))
    if has_errbars:
        ax1_evo.errorbar(indices, all_coeffs[:, 3],
                         yerr=delta_coeffs_waves[:, 3],
                         fmt='D-', color='black', label="Z4: Defocus", lw=3, capsize=4)
    else:
        ax1_evo.plot(indices, all_coeffs[:, 3], 'D-', color='black', label="Z4: Defocus", lw=3)
    ax1_evo.set_ylabel("Defocus [waves λ]", fontweight='bold')
    ax1_evo.tick_params(axis='y')
    
    ax2_evo = ax1_evo.twinx()
    colors_evo = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_evo = ["Astig 1 (Z5)", "Astig 2 (Z6)", "Coma 1 (Z7)", "Coma 2 (Z8)"]
    for j, col, lab in zip(range(4, 8), colors_evo, labels_evo):
        if j < all_coeffs.shape[1]:
            if has_errbars:
                ax2_evo.errorbar(indices, all_coeffs[:, j],
                                 yerr=delta_coeffs_waves[:, j],
                                 fmt='o--', color=col, label=lab, capsize=3)
            else:
                ax2_evo.plot(indices, all_coeffs[:, j], 'o--', color=col, label=lab)
    ax2_evo.set_ylabel("Shape Aberrations [waves λ]", color='blue')
    ax2_evo.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Aberration Evolution: Structure vs Propagation")
    ax1_evo.set_xticks(indices)
    ax1_evo.set_xticklabels(names, rotation=45, ha='right')
    ax1_evo.legend(loc="upper left")
    ax2_evo.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # Plot 2: Quality Graph
    fig_qual, ax_rms = plt.subplots(figsize=(10, 5))
    if has_errbars:
        ax_rms.errorbar(indices, rms_originale,
                        yerr=error_budget['delta_rms'],
                        fmt='s-', color='firebrick', label='Total RMS', capsize=4)
        ax_rms.errorbar(indices, rms_residua,
                        yerr=error_budget['delta_rms'] * 0.5,
                        fmt='o--', color='forestgreen', label='Residual RMS (Estimated)', capsize=4)
    else:
        ax_rms.plot(indices, rms_originale, 's-', color='firebrick', label='Total RMS')
        ax_rms.plot(indices, rms_residua, 'o--', color='forestgreen', label='Residual RMS (Estimated)')
    ax_rms.set_ylabel("RMS [waves λ]")
    ax_rms.grid(True, alpha=0.3)
    
    ax_s = ax_rms.twinx()
    if has_errbars:
        ax_s.bar(indices, strehl_ratio, alpha=0.3, color='gold',
                 yerr=error_budget['delta_strehl'], capsize=3, label='Strehl Ratio')
    else:
        ax_s.bar(indices, strehl_ratio, alpha=0.3, color='gold', label='Strehl Ratio')
    ax_s.set_ylim(0, 1.1)
    ax_s.set_ylabel("Strehl Ratio (1.0 = Ideal)", color='goldenrod')
    
    plt.title("Wavefront Quality and Fidelity Analysis")
    ax_rms.set_xticks(indices)
    ax_rms.set_xticklabels(names, rotation=45, ha='right')
    lines_q, labels_q = ax_rms.get_legend_handles_labels()
    lines2_q, labels2_q = ax_s.get_legend_handles_labels()
    ax_rms.legend(lines_q + lines2_q, labels_q + labels2_q, loc='center left')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
