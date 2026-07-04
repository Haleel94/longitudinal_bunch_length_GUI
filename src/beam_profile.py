
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

def gaussian_with_constant(x, A, mean, sigma, C):
    return A * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + C

def calculate_profiles(image, window_size):
    k = 1.26  # Scaling factor

    # Adaptive Slicing based on High-Intensity Region around Maximum Intensity
    max_pos = np.unravel_index(np.argmax(image), image.shape)

    # Calculate slicing bounds, ensuring they stay within the image dimensions
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(image.shape[1], max_pos[1] + window_size)

    # Extract adaptively sliced images
    sliced_img_hor = image[slice_start_row:slice_end_row, :]
    sliced_img_vert = image[:, slice_start_col:slice_end_col]

    # Compute profiles by summing intensities
    profile_sliced_hor = np.sum(sliced_img_hor, axis=0)
    profile_sliced_vert = np.sum(sliced_img_vert, axis=1)


    return sliced_img_hor, profile_sliced_hor, profile_sliced_vert

def plot_beam_profiles(image, ax_hor, ax_vert, window_size):

    #k = 1.26

    # Calculate profiles
    sliced_img_hor, profile_sliced_hor, profile_sliced_vert = calculate_profiles(image, window_size)

    #scale
    x_hor = np.arange(len(profile_sliced_hor))
    x_vert = np.arange(len(profile_sliced_vert))


    # Fit Gaussian to horizontal profile
    initial_guess_hor = [np.max(profile_sliced_hor), np.argmax(profile_sliced_hor), 1.0, np.min(profile_sliced_hor)]
    bounds_hor = (0, [np.inf, len(profile_sliced_hor), np.inf, np.inf])
    params_hor, _ = curve_fit(gaussian_with_constant, x_hor, profile_sliced_hor, p0=initial_guess_hor, bounds=bounds_hor, maxfev=10000)
    fwhm_hor = 2.355 * params_hor[2]  # FWHM calculation for horizontal

    # Fit Gaussian to vertical profile

    initial_guess_vert = [np.max(profile_sliced_vert), np.argmax(profile_sliced_vert), 1.0, np.min(profile_sliced_vert)]
    bounds_vert = (0, [np.inf, len(profile_sliced_vert), np.inf, np.inf])
    params_vert, _ = curve_fit(gaussian_with_constant, x_vert, profile_sliced_vert, p0=initial_guess_vert,
                               bounds=bounds_vert,  maxfev=10000)
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


