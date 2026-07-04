
import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

def gaussian_with_constant(x, A, mean, sigma, C):
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C

def calculate_profiles(image, window_size, angle):

    k = 1.26  # Scaling factor
    # Normalize image intensities to the 0-255 range
    img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_normalized = img_normalized[300:350]  # Extracting a region
    rows, cols = img_normalized.shape
    center = (cols // 2, rows // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Calculate new image dimensions based on the rotation angle
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])

    new_width = int(rows * sin_angle + cols * cos_angle)
    new_height = int(cols * sin_angle + rows * cos_angle)

    # Adjust the rotation matrix to take into account the translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate the image with the adjusted matrix
    rotated_image = cv2.warpAffine(img_normalized, rotation_matrix, (new_width, new_height))

    # Check if the rotated image is empty
    if rotated_image.size == 0:
        raise ValueError("Rotated image is empty. Check input dimensions and angle.")


    # Adaptive Slicing based on High-Intensity Region around Maximum Intensity
    max_pos = np.unravel_index(np.argmax(rotated_image), rotated_image.shape)

    # Calculate slicing bounds, ensuring they stay within the image dimensions
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(rotated_image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(image.shape[1], max_pos[1] + window_size)

    # Extract adaptively sliced images
    sliced_img_hor = rotated_image[slice_start_row:slice_end_row, :]
    sliced_img_vert = rotated_image[:, slice_start_col:slice_end_col]

    # Compute profiles by summing intensities
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)

    # Smooth profiles for Gaussian fitting
    smoothed_prof_hor = gaussian_filter(profile_sliced_hor, sigma=2)
    smoothed_prof_vert = gaussian_filter(profile_sliced_vert, sigma=2)

    return sliced_img_hor,   smoothed_prof_hor,  smoothed_prof_vert

def plot_smoothed_beam_profiles(image, ax_hor, ax_vert, window_size, angle_rotation):

    k = 1.26

    # Calculate profiles
    sliced_img_hor, smoothed_prof_hor,  smoothed_prof_vert = calculate_profiles(image, window_size, angle_rotation)

    #scale
    x_hor = np.arange(len(smoothed_prof_hor))
    x_vert = np.arange(len(smoothed_prof_vert))


    # Fit Gaussian to horizontal profile
    initial_guess_hor = [np.max(smoothed_prof_hor), np.argmax(smoothed_prof_hor), 1.0, np.min(smoothed_prof_hor)]
    bounds_hor = (0, [np.inf, len(smoothed_prof_hor), np.inf, np.inf])
    params_hor, _ = curve_fit(gaussian_with_constant, x_hor, smoothed_prof_hor, p0=initial_guess_hor, bounds=bounds_hor)
    fwhm_hor = 2.355 * params_hor[2]  # FWHM calculation for horizontal

    # Fit Gaussian to vertical profile

    initial_guess_vert = [np.max(smoothed_prof_vert), np.argmax(smoothed_prof_vert), 1.0, np.min(smoothed_prof_vert)]
    bounds_vert = (0, [np.inf, len(smoothed_prof_vert), np.inf, np.inf])
    params_vert, _ = curve_fit(gaussian_with_constant, x_vert, smoothed_prof_vert, p0=initial_guess_vert,
                               bounds=bounds_vert)
    fwhm_vert = 2.355 * params_vert[2]  # FWHM calculation for vertical

    ax_hor.clear()
    ax_vert.clear()

    #plot the profiles
    ax_hor.plot(x_hor, smoothed_prof_hor, label='Горизонтальный профиль', color='blue')
    ax_hor.plot(x_hor, gaussian_with_constant(x_hor, *params_hor), 'r-', label=f'Аппроксимация(FWHM={fwhm_hor:.2f})')
    ax_hor.set_xlabel('пс')
    ax_hor.set_ylabel('Интенсивность')
    ax_hor.legend()

    ax_vert.plot(smoothed_prof_vert, x_vert, label='Вертикальный профиль', color='blue')
    ax_vert.plot(gaussian_with_constant(x_vert, *params_vert), x_vert, 'r-', label=f'Аппроксимация(FWHM={fwhm_vert:.2f})')
    ax_vert.set_ylabel('пс')
    ax_vert.set_xlabel('Интенсивность')
    ax_vert.legend()


    return sliced_img_hor, params_hor, params_vert


