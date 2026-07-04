# Longitudinal_bunch_length_GUI_project
# Longitudinal_bunch_length_Analysis_GUI

This repository contains a GUI application built with PyQt5, designed to calculate the beam profile from images captured with a streak camera at VEPP-4M, NovoFEL, and VEPP-2000. The tool allows users to visualize images, analyze beam profiles, and generate data plots to study the energy spread and beam characteristics based on synchrotron radiation data.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The Beam Profile Analysis GUI provides an intuitive interface for beam diagnostics by leveraging streak camera images captured at Russian research facilities, including VEPP-4M, NovoFEL, and VEPP-2000. Using these images, the tool extracts the beam profile and calculates key parameters such as beam length and distance between micro bunches. The GUI simplifies complex operations, offering researchers a flexible and interactive platform for real-time data visualization and analysis.

### Background

- **Facilities**: VEPP-4M, NovoFEL, VEPP-2000, Microtron facility at Tomsk polytechnic and any other facility
- **Techniques**: Streak camera imaging, PyQt5 for GUI, and synchrotron radiation analysis
- **Purpose**: Real time measurement of longitudinal bunch length

## Features

- **Image Import**: Supports standard image file formats and `.npy` files for compatibility with diverse data sources.
- **Image Visualization**: Display images of streak camera captures with adjustable contrast for faint or dark images.
- **Beam Profile Calculation**: Plot both vertical and horizontal beam profiles, yielding Gaussian-like curves to determine beam width and other parameters.
- **File-Based Analysis**: Handles multiple formats, such as `.png`, `.jpg`, and `.npy`, ensuring flexibility in input file handling.
- **Webcam Integration**: Connects to a webcam for image capture, allowing real-time visualization and analysis.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/beam-profile-gui.git
   cd beam-profile-gui
   ```

2. **Set Up the Virtual Environment** (optional but recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the GUI Application**
   ```bash
   python main.py
   ```

## Usage

1. Launch the application using `python main.py`.
2. Use the file upload feature to select a streak camera image.
4. Adjust visualization settings for faint images.
5. View the calculated beam profile, which can be saved or further analyzed.

## File Structure

```plaintext
longitudinal_bunch_length_gui/
├── src/
│   ├── main.py                         # Main GUI application
│   ├── beam_profile.py                 # Beam profile extraction from streak camera images
│   ├── beam_profile_npy_files.py       # Beam profile analysis for .npy data files
│   ├── FWHM_cal.py                     # Full Width at Half Maximum (FWHM) calculations
│   ├── distance_btw_micro_bunches.py   # Micro-bunch spacing analysis
│   ├── noise_smoothing.py              # Noise reduction and smoothing utilities
│   └── phosphor_screen_calib.py        # Phosphor screen calibration functions
├── examples/                           # Example input files and sample data
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

## Contributing

Contributions are welcome! If you’d like to add new features or improve existing ones, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thank you for using the Beam Profile Analysis GUI! For any issues or suggestions, feel free to open an issue on GitHub or reach out.
