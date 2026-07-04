import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd


def fwhm_profile(image, window_size):
    max_value = np.max(image)
    max_pos = np.unravel_index(np.argmax(image), image.shape)

    # Extract ROI
    slice_start_row = max(0, max_pos[0] - window_size)
    slice_end_row = min(image.shape[0], max_pos[0] + window_size)
    slice_start_col = max(0, max_pos[1] - window_size)
    slice_end_col = min(image.shape[1], max_pos[1] + window_size)

    signal_h = image[slice_start_row:slice_end_row, :]
    signal_hor = np.sum(signal_h, axis=0)
    signal_hor_filtered = nd.median_filter(signal_hor, size=20)

    signal_v = image[:, slice_start_col:slice_end_col]
    signal_vert = np.sum(signal_v, axis=1)
    signal_vert_filtered = nd.median_filter(signal_vert, size=20)

    # Half-maximum values
    half_width_max_hor = np.max(signal_hor_filtered) / 2
    half_width_max_vert = np.max(signal_vert_filtered) / 2

    # Find indices at half-maximum
    indices_above_half_max_hor = np.where(signal_hor_filtered > half_width_max_hor)[0]
    index_left_hor = indices_above_half_max_hor[0]
    index_right_hor = indices_above_half_max_hor[-1]

    indices_above_half_max_vert = np.where(signal_vert_filtered > half_width_max_vert)[0]
    index_left_vert = indices_above_half_max_vert[0]
    index_right_vert = indices_above_half_max_vert[-1]

    # Group all outputs into a dictionary
    return {
        "signal_h": signal_h,
        "signal_hor": signal_hor,
        "signal_hor_filtered": signal_hor_filtered,
        "signal_v": signal_v,
        "signal_vert": signal_vert,
        "signal_vert_filtered": signal_vert_filtered,
        "half_width_max_hor": half_width_max_hor,
        "half_width_max_vert": half_width_max_vert,
        "index_left_hor": index_left_hor,
        "index_right_hor": index_right_hor,
        "index_left_vert": index_left_vert,
        "index_right_vert": index_right_vert
    }


def plot_fwhm_profile(image, ax_hor, ax_vert, window_size):
    # Extract the profiles and relevant data
    profile_data = fwhm_profile(image, window_size)

    # Unpack the dictionary for plotting
    signal_h = profile_data["signal_h"]
    signal_hor = profile_data["signal_hor"]
    signal_hor_filtered = profile_data["signal_hor_filtered"]
    signal_v = profile_data["signal_v"]
    signal_vert = profile_data["signal_vert"]
    signal_vert_filtered = profile_data["signal_vert_filtered"]
    half_width_max_hor = profile_data["half_width_max_hor"]
    half_width_max_vert = profile_data["half_width_max_vert"]
    index_left_hor = profile_data["index_left_hor"]
    index_right_hor = profile_data["index_right_hor"]
    index_left_vert = profile_data["index_left_vert"]
    index_right_vert = profile_data["index_right_vert"]

    A = index_right_hor
    B = index_left_hor
    FWHM = A - B

    x_hor = np.arange(len(signal_hor))
    x_vert = np.arange(len(signal_vert))

    # Clear the axes before plotting
    ax_hor.clear()
    ax_vert.clear()

    # Horizontal profile
    ax_hor.plot(x_hor, signal_hor, label='Горизонтальный профиль', linewidth=1)
    line_length = 0.3 * np.max(signal_hor_filtered)
    ax_hor.plot(x_hor, signal_hor_filtered, color='orange', label=f'Аппроксимация(FWHM={FWHM:.2f})', linestyle='-')
    ax_hor.vlines(x=index_left_hor, ymin=half_width_max_hor - line_length, ymax=half_width_max_hor + line_length,
            color='red', linestyle='-')
    ax_hor.vlines(x=index_right_hor, ymin=half_width_max_hor - line_length, ymax=half_width_max_hor + line_length,
                  color='green', linestyle='-')
    ax_hor.legend()


    # Vertical profile
    ax_vert.plot(signal_vert, x_vert, label='Вертикальный профиль', linewidth=1)
    ax_vert.plot(signal_vert_filtered, x_vert, color='orange', label='Signal Filtered', linestyle='-')
    line_length = 0.3 * np.max(signal_vert_filtered)
    ax_vert.hlines(y=index_left_vert, xmin=half_width_max_vert - line_length, xmax=half_width_max_vert + line_length,
                   color='red', linestyle='-')
    ax_vert.hlines(y=index_right_vert, xmin=half_width_max_vert - line_length, xmax=half_width_max_vert + line_length,
                   color='green', linestyle='-')
    ax_vert.legend()


    return profile_data
