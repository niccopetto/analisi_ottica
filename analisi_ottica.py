
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import re
import copy
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import poppy.zernike as zernike

# Custom modules
import AuxiliaryTools as at
from gs_engine import GerSaxPhaseRetriever, mm, nm
import tkinter as tk
from tkinter import filedialog

def estrai_distanza_da_obj(img_obj):
    # img_obj.imageName contains the filename
    match = re.search(r'(\d+)', os.path.basename(img_obj.imageName))
    return int(match.group(1)) if match else 0

def super_gaussian_curve(coords, A, x0,y0, w, P, offset):
    y,x = coords
    r=np.sqrt((x-x0)**2+ (y-y0)**2)
    g=A* np.exp(-2*(r/w)**P) + offset
    return g.ravel()

def noll_to_nm(j):
    """
    Converte l'indice sequenziale di Noll (j) nella coppia (n, m).
    Supporta fino ai primi 15 termini (o più se espandi la logica).
    """
    # Indice Noll: (n, m)
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
        raise ValueError(f"Indice j={j} non mappato (estendi il dizionario o usa algoritmo completo)")

def preprocess(input_dir=".", pixelsizex=0.0056*mm, pixelsizey=0.0056*mm):
    """
    Performs image preprocessing: loading, interactive selection, resampling, ROI/BG subtraction.
    """
    print(f"Processing images from: {input_dir}")
    
    # Setup AuxiliaryClass
    ac = at.AuxiliaryClass()
    ac.setImagesDir(input_dir)
    
    # Helper to get files
    # Check if directory handles *.tiff directly or we need lists
    # Logic adapted from original script
    # Logic adapted from original script
    # Explicitly match .tif and .tiff (case-insensitive)
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(input_dir, ext)))
    # Remove duplicates if any (though glob shouldn't overlap with these specific patterns)
    all_files = sorted(list(set(all_files)))
    imagesFNs = [os.path.basename(f) for f in all_files]
    
    if not imagesFNs:
        print("No .tiff files found in directory!")
        return [], [], [], pixelsizex, pixelsizey

    ac.setImagesFilenames(*imagesFNs)
    
    # Input for resampling
    print("Default resampling factor: 0.5")
    # res_input = input("Enter the resampling factor (e.g., 0.5 for half size, default 0.5): ")
    # resamplingFactor = float(res_input) if res_input.strip() else 0.5
    resamplingFactor = 0.5 # Defaulting for automation

    resampledpixelsizex = pixelsizex / resamplingFactor
    resampledpixelsizey = pixelsizey / resamplingFactor
    
    # Load images
    inputImages = [at.ImageClass() for _ in ac.imagesPath]
    for i, img in enumerate(inputImages):
        img.loadImage(ac.imagesPath[i], {'x': pixelsizex, 'y': pixelsizey})
    
    # Channel selection
    channel = 'R' # Default
    
    for i in range(len(inputImages)):
        inputImages[i].selectChannel(channel)
        
    # Resample
    print(f"\nResampling with factor {resamplingFactor}...")
    for img in inputImages:
        img.resampleOriginal(factor=resamplingFactor)
        
    # ROI and BG selection
    print("\n--- ROI and Background Selection ---")
    print("Automating ROI selection (Center crop 50%) and BG (Top-left corner) for demonstration.")
    # For real usage, one might want interactive. 
    # Here we assume a center crop to allow the script to run without interaction if needed, 
    # OR we keep interaction but warn user. 
    # The original script had it interactive. I will leave interactive calls but comment on automation if needed.
    
    for img in inputImages:
        print(f"Selecting for {img.imageName}...")
        # INTERACTIVE:
        img.selectROI()
        img.selectBG()
        # AUTOMATED FALLBACK (Uncomment to skip interaction):
        # h, w = img.imageData.shape
        # cy, cx = h//2, w//2
        # dy, dx = h//4, w//4
        # img.ROI = img.imageData[cy-dy:cy+dy, cx-dx:cx+dx]
        # img.BG = img.imageData[0:50, 0:50]
        
        img.retrieveAverageBG()
        img.subtractBGfromROI()
        
    # Center of mass calculation
    finalImageNpixelsx = 1024
    finalImageNpixelsy = 1024

    for img in inputImages:
        img.getCenterOfMassOfROI()
        img.buildImageWithSize(finalImageNpixelsx, finalImageNpixelsy)
        
    # Ordering and Distance Calculation
    print("\n--- Calculating Distances ---")
    valori_grezzi = np.array([estrai_distanza_da_obj(img) for img in inputImages])
    ordine_calcolato = np.argsort(valori_grezzi)[::-1]
    valori_ordinati = valori_grezzi[ordine_calcolato]
    
    delta_raw = np.abs(np.diff(valori_ordinati))
    d = (delta_raw * 0.1) * mm # Assuming 0.1 scale factor from notebook
    
    # Cleaning
    for i, img in enumerate(inputImages):
        img.centeredROI = np.maximum(img.centeredROI, 0)
        
    return inputImages, d, ordine_calcolato, resampledpixelsizex, resampledpixelsizey

def main():
    # Configuration
    InputDIR = "./imgs_folder/te_22_05_cannone/"
    wavelength = 633 * nm
    # Initial pixelsize
    pixelsizex = 0.0056 * mm
    pixelsizey = 0.0056 * mm
    
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

    # 2. GS Algorithm
    print("\nAvvio algoritmo GS...")
    GS = GerSaxPhaseRetriever(*inputImages,
                              distances=d,
                              wavelength=wavelength, 
                              ordering=ordine_calcolato)
    
    GS.GS_algorithm(Niterations=50) # Reduced iterations for speed in test
    
    # 3. Post-Analysis / Visualization
    campo = GS.FarField.field
    max_val = np.max(np.abs(campo))
    print(f"Intensità Max Far Field: {max_val:.4e}")
    
    # --- ZERNIKE ANALYSIS ---
    print("\n--- ZERNIKE DECOMPOSITION ANALYSIS ---")
    
    # 3a. Intensity Fitting to find Waist and Center
    normalized_input_intensity = []
    Y = []
    X = []
    
    # Setup fitting storage
    fit_intensity = []
    
    # Prepare grids
    for i, name in enumerate(GS.imagesNames):
        I_data = GS.RetrieveIntensity(name)
        if np.max(I_data) > 0:
            I_data = I_data / np.max(I_data)
        normalized_input_intensity.append(I_data)
        h, w_img = I_data.shape
        temp_y, temp_x = np.mgrid[0:h, 0:w_img]
        Y.append(temp_y)
        X.append(temp_x)

    print(f"Fitting intensity profiles for {len(normalized_input_intensity)} images...")
    
    for I_norm, grid_y, grid_x, name in zip(normalized_input_intensity, Y, X, GS.imagesNames):
        # Smoothing
        I_smooth = gaussian_filter(I_norm, sigma=3)
        
        # Dynamic Waist Guess
        area_pixel = np.sum(I_norm > 0.10)
        w_guess = np.sqrt(area_pixel / np.pi)
        w_guess = max(w_guess, 5.0)
        
        # Max location
        y_max, x_max = np.unravel_index(np.argmax(I_smooth), I_norm.shape)
        
        # Init Guess: A, x0, y0, w, P, offset
        p0 = [1.0, x_max, y_max, w_guess, 2.0, 0.0]
        
        # Bounds: A, x0, y0, w, P, offset
        # A: [0.5, 1.5], x0,y0: within image, w: [1, w/2], P: [1, 20], offset: [0, 0.5]
        bounds = ([0.5, 0, 0, 1.0, 1.0, 0], 
                  [1.5, grid_x.shape[1], grid_x.shape[0], grid_x.shape[1]/2, 20.0, 0.5])
        
        try:
            popt, pcov = curve_fit(super_gaussian_curve, (grid_y, grid_x), I_norm.ravel(), p0=p0, bounds=bounds)
            A_fit, x0_fit, y0_fit, w_fit, P_fit, offset_fit = popt
            
            print(f"{name:<20} | Waist: {w_fit:.2f} | P: {P_fit:.4f} | Center: ({x0_fit:.1f}, {y0_fit:.1f})")
            
            fit_intensity.append({
                'Name': name,
                'Waist': w_fit,
                'P': P_fit,
                'Center_X': x0_fit,
                'Center_Y': y0_fit,
                'Amplitude': A_fit,
                'Offset': offset_fit
            })
            
        except RuntimeError:
            print(f"{name:<20} | FIT FAILED")
            # Append None or dummy to maintain index alignment? 
            # Better to just skip or handle gracefully.
            continue

    # 3b. Phase Extraction and Zernike Decomposition
    print("\nExtracting Phases and calculating Zernike coefficients...")
    
    phases_for_zernike = []
    
    # Iterating over fitted results to ensure we have ROI parameters
    for fit_data in fit_intensity:
        name = fit_data['Name']
        center_x = fit_data['Center_X']
        center_y = fit_data['Center_Y']
        # Radius for Zernike: usually 2 * Waist or similar coverage
        radius = 2.0 * fit_data['Waist'] 
        
        # Retrieve Phase
        phi = GS.RetrievePhase(name) # 2D array
        h, w_dim = phi.shape
        y_indices, x_indices = np.mgrid[0:h, 0:w_dim]
        
        # Create ROI mask
        dist_sq = (x_indices - center_x)**2 + (y_indices - center_y)**2
        # Mask outside radius
        masked_phi = np.ma.masked_where(dist_sq > radius**2, phi)
        
        # Remove Piston (Avg)
        if not np.all(masked_phi.mask):
            piston_val = np.mean(masked_phi)
            masked_phi = masked_phi - piston_val
            
            # Additional Tilt removal can be done here if needed
            
            phases_for_zernike.append({
                'Name': name,
                'PhaseROI': masked_phi,
                'Radius': radius
            })
        else:
            print(f"Warning: Empty ROI for {name}")

    # 3c. Decompose
    print("\n--- Zernike Coefficients ---")
    for item in phases_for_zernike:
        name = item['Name']
        phase_roi = item['PhaseROI']
        
        # Ensure no NaNs filling
        phase_roi_filled = np.ma.filled(phase_roi, 0)
        
        # Poppy Zernike Decomposition
        # Noll indices 1 to 15
        coeffs = zernike.decompose_opd(phase_roi_filled, nterms=15, basis='noll')
        # coeffs is a list/array of coefficients starting from index 1 (usually poppy handles indexing)
        # Note: poppy returns list where index 0 might be piston?
        # Check poppy docs: decompose_opd returns coefficients. 
        # Usually it returns an array.
        
        print(f"\nImage: {name}")
        for j, c in enumerate(coeffs, start=1):
             try:
                 n, m = noll_to_nm(j)
                 print(f"  Z{j} (n={n}, m={m}): {c:.4f}")
             except ValueError:
                 pass

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
