import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def gaussian_with_constant(x, A, mean, sigma, C):
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C


def calculate_npy_profile(image, window_size, angle):
    # Normalize image intensities to the 0-255 range
    img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_normalized = img_normalized[300:350]  # Extracting a region
    rows, cols = img_normalized.shape
    center = (cols // 2, rows // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Calculate new image dimensions
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int(rows * sin_angle + cols * cos_angle)
    new_height = int(cols * sin_angle + rows * cos_angle)

    # Adjust the rotation matrix for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate the image
    rotated_image = cv2.warpAffine(img_normalized, rotation_matrix, (new_width, new_height))

    if rotated_image.size == 0:
        raise ValueError("Rotated image is empty. Check input dimensions and angle.")

    # Find the position of maximum intensity
    max_pos = np.unravel_index(np.argmax(rotated_image), rotated_image.shape)

    # Define slicing bounds
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(rotated_image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(rotated_image.shape[1], max_pos[1] + window_size)

    # Slice images horizontally and vertically
    sliced_img_hor = rotated_image[slice_start_row:slice_end_row, :]
    sliced_img_vert = rotated_image[:, slice_start_col:slice_end_col]

    if sliced_img_hor.size == 0 or sliced_img_vert.size == 0:
        raise ValueError("Sliced image region is empty. Adjust window size.")

    # Compute intensity profiles
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)

    return sliced_img_hor, profile_sliced_hor, profile_sliced_vert


def plot_beam_profiles_npy(image, ax_hor, ax_vert, window_size, angle_rotation):

    k = 1.26

    # Calculate profiles
    sliced_img_hor, profile_sliced_hor, profile_sliced_vert = calculate_npy_profile(image, window_size, angle_rotation)

    #scale
    x_hor = np.arange(len(profile_sliced_hor)) * k
    x_vert = np.arange(len(profile_sliced_vert)) * k


    # Fit Gaussian to horizontal profile
    initial_guess_hor = [np.max(profile_sliced_hor), np.argmax(profile_sliced_hor), 1.0, np.min(profile_sliced_hor)]
    bounds_hor = (0, [np.inf, len(profile_sliced_hor), np.inf, np.inf])
    params_hor, _ = curve_fit(gaussian_with_constant, x_hor, profile_sliced_hor, p0=initial_guess_hor, bounds=bounds_hor)
    fwhm_hor = 2.355 * params_hor[2]  # FWHM calculation for horizontal

    # Fit Gaussian to vertical profile

    initial_guess_vert = [np.max(profile_sliced_vert), np.argmax(profile_sliced_vert), 1.0, np.min(profile_sliced_vert)]
    bounds_vert = (0, [np.inf, len(profile_sliced_vert), np.inf, np.inf])
    params_vert, _ = curve_fit(gaussian_with_constant, x_vert, profile_sliced_vert, p0=initial_guess_vert,
                               bounds=bounds_vert)
    fwhm_vert = 2.355 * params_vert[2]  # FWHM calculation for vertical

    ax_hor.clear()
    ax_vert.clear()

    #plot the profiles
    ax_hor.plot(x_hor, profile_sliced_hor, label='Горизонтальный профиль', color='blue')
    ax_hor.plot(x_hor, gaussian_with_constant(x_hor, *params_hor), 'r-', label=f'Аппроксимация(FWHM={fwhm_hor:.2f})')
    ax_hor.set_xlabel('пс')
    ax_hor.set_ylabel('Интенсивность')
    ax_hor.legend()

    ax_vert.plot(profile_sliced_vert, x_vert, label='Вертикальный профиль', color='blue')
    ax_vert.plot(gaussian_with_constant(x_vert, *params_vert), x_vert, 'r-', label=f'Аппроксимация(FWHM={fwhm_vert:.2f})')
    ax_vert.set_ylabel('пс')
    ax_vert.set_xlabel('Интенсивность')
    ax_vert.legend()


    return sliced_img_hor, params_hor, params_vert
