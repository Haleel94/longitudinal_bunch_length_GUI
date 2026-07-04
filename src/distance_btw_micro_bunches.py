import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

def gaussian_with_constant(x, A, mean, sigma, C):
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C

def electron_beam_train_profile(image, window_size):

    # Load multiple background images
    background_images = [
        np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_0.npy'),
        np.load('D:/project/Data/save/background_x=-1.78mA_z=-2.55kV_1.npy'),
        np.load('D:/project/Data/save/background_x=-1.79mA_z=-2.55kV_2.npy'),
        np.load('D:/project/Data/save/background_x=-1.79mA_z=-2.55kV_3.npy'),
        np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_4.npy'),
        np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_5.npy')
    ]

    # Compute the average background
    bck_avrg = np.mean(background_images, axis=0).astype(np.uint8)

    # Convert images to OpenCV format (8-bit grayscale)
    img_cv = image.astype(np.uint8)

    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # Apply background subtraction
    bg_subtractor.apply(bck_avrg)  # Learn background first
    fg_mask = bg_subtractor.apply(img_cv)  # Apply on main image

    # Extract the foreground and apply Gaussian blur
    foreground = cv2.bitwise_and(img_cv, img_cv, mask=fg_mask)
    image = foreground[350:550]

    # Find the global maximum
    max_pos = np.unravel_index(np.argmax(image), image.shape)

    # Define slicing window
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(image.shape[1], max_pos[1] + window_size)

    sliced_img_hor = image[slice_start_row:slice_end_row, :]
    sliced_img_vert = image[:, slice_start_col:slice_end_col]

    # Compute profiles by summing intensities
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)


    # Find peaks in the sliced signal



    return sliced_img_hor, profile_sliced_hor, profile_sliced_vert



def electron_beam_train_plots(image, ax_hor, ax_vert, window_size, peak_distance):

    sliced_img_hor, profile_sliced_hor, profile_sliced_vert = electron_beam_train_profile(image, window_size)

    peaks, _ = find_peaks(profile_sliced_hor, height=max(profile_sliced_hor) * 0.2, distance=peak_distance)

    if len(peaks) >= 2:
        peak_distances = np.diff(peaks)
        avg_distance = np.mean(peak_distances)
    else:
        avg_distance = 0

    window = int(np.mean(np.diff(peaks)) // 2)

    # To store popt results for each peak
    popt_all = []

    x = np.arange(len(profile_sliced_hor))

    for i, peak in enumerate(peaks):
        start = max(0, peak - window)
        end = min(len(profile_sliced_hor) - 1, peak + window)

        x_local = x[start:end]
        y_local = profile_sliced_hor[start:end]

        initial_guess = [max(y_local), peak, 20, np.min(y_local)]  # Fix: max(y_local) instead of max(x_local)
        try:
            popt, _ = curve_fit(gaussian_with_constant, x_local, y_local, p0=initial_guess)
            popt_all.append((start, end, popt))  # Store fit parameters along with their range
        except RuntimeError:
            print(f"Fit failed for peak at index {peak}")

# Calculate distances between peaks

    ax_hor.clear()
    ax_vert.clear()

    ax_hor.set_title("Sliced Signal and Gaussian Fit")
    ax_hor.plot(profile_sliced_hor, linewidth=1, label='Original Signal', color='blue')

    first = True
    for (start, end, popt) in popt_all:
        x_fit = np.arange(start, end)  # Fix: Generate correct x values
        y_fit = gaussian_with_constant(x_fit, *popt)
        ax_hor.plot(x_fit, y_fit, color='r', linestyle='-', label=f'Gaussian Fit (Peak at {popt[1]:.1f})')
        ax_hor.axvline(start, color='y', linestyle='--', label='Fit Window Start' if start == popt_all[0][0] else "")
        ax_hor.axvline(end, color='g', linestyle='--', label='Fit Window End' if start == popt_all[0][0] else "")
        ax_hor.axvspan(start, end, color='yellow', alpha=0.2)
        first = False
    # Annotate distances between peaks

    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            ax_hor.text((peaks[i] + peaks[i + 1]) / 2, max(profile_sliced_hor) * 0.5,
                 f"{peaks[i + 1] - peaks[i]} px", color='black', fontsize=10)

    # Annotate sigma values for the Gaussian fits

    sigma_values = []

    for i, (start, end, popt) in enumerate(popt_all):
        sigma_value = np.round(popt[2], 2)  # Extract sigma from Gaussian fit
        sigma_values.append(sigma_value)

        if i < len(peaks):
            ax_hor.text(peaks[i], max(profile_sliced_hor) * 0.5, f"σ={sigma_value}",
                 color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Compute the average sigma
    avg_sigma = np.mean(sigma_values) if sigma_values else 0

    # Annotate the average sigma
    ax_hor.text(len(profile_sliced_hor) * 0.5, max(profile_sliced_hor) * 0.6,
         f"Avg σ: {avg_sigma:.2f}",
         fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.8))

    # Annotate average distance
    plt.text(len(profile_sliced_hor) * 0.5, max(profile_sliced_hor) * 0.8,
         f"Avg Distance: {avg_distance:.2f} px",
         fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.8))

    # for pos in peaks:
    # plt.axvline(pos, color='y', linestyle='--', label='Peak' if pos == peaks[0] else "")

    x_vert = np.arange(len(profile_sliced_vert))
    initial_guess_vert = [np.max(profile_sliced_vert), np.argmax(profile_sliced_vert), 1.0, np.min(profile_sliced_vert)]
    bounds_vert = (0, [np.inf, len(profile_sliced_vert), np.inf, np.inf])
    params_vert, _ = curve_fit(gaussian_with_constant, x_vert, profile_sliced_vert, p0=initial_guess_vert,
                               bounds=bounds_vert, maxfev=10000)
    fwhm_vert = 2.355 * params_vert[2]  # FWHM calculation for vertical


    ax_vert.clear()

    ax_vert.plot(profile_sliced_vert, x_vert, label='Вертикальный профиль', color='blue')
    ax_vert.plot(gaussian_with_constant(x_vert, *params_vert), x_vert, 'r-',
                 label=f'Аппроксимация(FWHM={fwhm_vert:.2f})')
    ax_vert.set_ylabel('пс')
    ax_vert.set_xlabel('Интенсивность')
    ax_vert.legend()

    return sliced_img_hor, popt_all, profile_sliced_vert, params_vert

