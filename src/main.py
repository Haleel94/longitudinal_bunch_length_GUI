import sys
import cv2
import numpy as np
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QSizePolicy, QDoubleSpinBox, QSpinBox, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from beam_profile import plot_beam_profiles
from beam_profile_npy_files import plot_beam_profiles_npy
from noise_smoothing import plot_smoothed_beam_profiles
from FWHM_cal import plot_fwhm_profile
from distance_btw_micro_bunches import electron_beam_train_profile, electron_beam_train_plots
class BeamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Profile Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # GUI Components
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(460, 280)

        # Buttons Layout
        self.button_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture Image", self)
        self.open_button = QPushButton("Open Image", self)
        self.profile_button = QPushButton("Plot Beam Profile", self)
        #self.save_button = QPushButton("Save results", self)

        self.button_layout.addWidget(self.capture_button)
        self.button_layout.addWidget(self.open_button)
        self.button_layout.addWidget(self.profile_button)
       #self.button_layout.addWidget(self.save_button)

        # Parameters Layout
        self.params_layout_hor = QHBoxLayout()
        self._add_param_label("Mean_hor:", self.params_layout_hor)
        self._add_param_label("Sigma_hor:", self.params_layout_hor)
        self._add_param_label("FWHM_hor:", self.params_layout_hor)

        # Parameters for vertical profiles
        self.params_layout_vert = QHBoxLayout()
        self._add_param_label("Mean_vert:", self.params_layout_vert)
        self._add_param_label("Sigma_vert:", self.params_layout_vert)
        self._add_param_label("FWHM_vert:", self.params_layout_vert)

        # window size label
        self.window_size_layout = QHBoxLayout()
        self.window_size_label = QLabel('Window Size', self)
        self.window_size_layout.addWidget(self.window_size_label)

        # window size spin box
        self.window_size_box = QSpinBox(self)
        self.window_size_box.setMinimum(20)
        self.window_size_box.setMaximum(1200)
        self.window_size_box.setValue(0)
        self.window_size_box.valueChanged.connect(self.value_change)
        self.window_size_layout.addWidget(self.window_size_box)

        # Angle layout
        self.angle_rotation_layout = QHBoxLayout()
        self.angle_rotation_label = QLabel("Angle", self)  # Corrected label text
        self.angle_rotation_layout.addWidget(self.angle_rotation_label)

        # Angle spin box
        self.angle_rotation_box = QDoubleSpinBox(self)
        self.angle_rotation_box.setMinimum(-30.0)  # Setting the range for rotation
        self.angle_rotation_box.setMaximum(30.0)
        self.angle_rotation_box.setSingleStep(0.1)
        self.angle_rotation_box.valueChanged.connect(self.angle_change)
        self.angle_rotation_layout.addWidget(self.angle_rotation_box)  # Corrected to addWidget

        #Peak distance layout
        self.peak_distance_layout = QHBoxLayout()
        self.peak_distance_label = QLabel('Peak Distance', self)
        self.peak_distance_layout.addWidget(self.peak_distance_label)

        #Peak Distance
        self.peak_distance_box = QDoubleSpinBox(self)
        self.peak_distance_box.setMinimum(0)
        self.peak_distance_box.setMaximum(1000)
        self.peak_distance_box.setValue(120)
        self.peak_distance_box.valueChanged.connect(self.peak_distance_change)
        self.peak_distance_layout.addWidget(self.peak_distance_box)

        # Background subtraction checkbox
        self.bg_subtraction_checkbox = QCheckBox(self)
        self.bg_subtraction_checkbox.setText('Background')
        self.bg_subtraction_checkbox.setChecked(False)
        self.bg_subtraction_checkbox.stateChanged.connect(self.bg_subtraction_state_change)
        self.button_layout.addWidget(self.bg_subtraction_checkbox)

        # Remove Noise checkbox
        self.guassian_filter_checkbox = QCheckBox(self)
        self.guassian_filter_checkbox.setText('Noise')
        self.guassian_filter_checkbox.setChecked(False)
        self.guassian_filter_checkbox.stateChanged.connect(self.guassian_filter_state_change)
        self.button_layout.addWidget(self.guassian_filter_checkbox)

        # FWHM profile checkbox
        self.fwhm_checkbox = QCheckBox(self)
        self.fwhm_checkbox.setText('FWHM')
        self.fwhm_checkbox.setChecked(False)
        self.fwhm_checkbox.stateChanged.connect(self.fwhm_state_change)
        self.button_layout.addWidget(self.fwhm_checkbox)

        #Electron Train
        self.electron_train_checkbox = QCheckBox("Enable Electron Train Analysis")
        self.electron_train_checkbox.setChecked(False)
        self.electron_train_checkbox.stateChanged.connect(self.electron_train_change)
        self.button_layout.addWidget(self.electron_train_checkbox)

        # Horizontal and Vertical Profile Canvases
        self.figure_hor, self.ax_hor = plt.subplots()
        self.canvas_hor = FigureCanvas(self.figure_hor)
        self.canvas_hor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas_hor.setFixedHeight(280)

        self.figure_vert, self.ax_vert = plt.subplots()
        self.canvas_vert = FigureCanvas(self.figure_vert)
        self.canvas_vert.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.canvas_vert.setFixedWidth(460)

        # Image Layout with Profiles
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_label)
        self.image_layout.addWidget(self.canvas_vert)

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.params_layout_hor)
        self.main_layout.addLayout(self.params_layout_vert)
        self.main_layout.addLayout(self.window_size_layout)
        self.main_layout.addLayout(self.angle_rotation_layout)
        self.main_layout.addLayout(self.peak_distance_layout)
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addWidget(self.canvas_hor)

        #Real_time updates of plots
        self.window_size_box.valueChanged.connect(self.plot_beam_profile)
        self.angle_rotation_box.valueChanged.connect(self.plot_beam_profile)

        # Set the main layout to the central widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Connect buttons to actions
        self.capture_button.clicked.connect(self.capture_image)
        self.open_button.clicked.connect(self.load_image)
        self.profile_button.clicked.connect(self.plot_beam_profile)
        #self.save_button.clicked.connect(self.save_results)

        self.captured_image = None
        self.loaded_image = None
        self.file_path = None  # Store file path for profile plotting

    def _add_param_label(self, text, layout):
        label = QLabel(text, self)
        layout.addWidget(label)
        value_label = QLabel(self)
        layout.addWidget(value_label)
        setattr(self, f"{text.lower().replace(':', '').strip()}_value", value_label)

    def value_change(self):
        value = self.window_size_box.value()
        #self.window_size_label.setText(f"Window size: {value}")  # Correct label reference

    def angle_change(self):
        value = self.angle_rotation_box.value()
        #self.angle_rotation_label.setText(f"Angle: {value}")  # Correct label reference

    def peak_distance_change(self):
       value = self.peak_distance_box.value()

    def safe_uncheck_checkbox(self):
        self.bg_subtraction_checkbox.blockSignals(True)
        self.bg_subtraction_checkbox.setChecked(False)
        self.bg_subtraction_checkbox.blockSignals(False)

    def fwhm_state_change(self, state):
        if state == 2:
            print("FWHM checkbox checked.")
            # Perform actions related to enabling FWHM (if needed)
        elif state == 0:  # Unchecked
            print("FWHM checkbox unchecked.")
    def electron_train_change(self, state):
        if state == 2:
            print('loaded')
        elif state == 0:
            print('')
    def guassian_filter_state_change(self, state):
        if state == 2:
            print('Noise checked')
        elif state == 0:
            print('Noise unchceked')

        self.plot_beam_profile()

    def capture_image(self):
        """Capture an image from the webcam."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.display_image(self.captured_image)

    #def save_results(self):
        #if self.loaded_image is None:
            #QMessageBox.warning(self, 'Warning', 'No processed image to save')
            #return
        #path, _ = QFileDialog.getSaveFileName(self, 'Save processed Image', '', 'Image Files(*.npy, *.pgm, *.jpg, *.png')
        #if path:
            #cv2.imwrite(path, self.loaded_image)

            #self.figure_hor.savefig('.png', '_hor.png')
            #QMessageBox.information(self, 'Saved', f'results saved to {path}')

    def display_image(self, sliced_img_hor):
        """Display the sliced image in the GUI using the viridis colormap."""
        if isinstance(sliced_img_hor, np.ndarray) and len(sliced_img_hor.shape) == 2:
            # Normalize the image to [0, 1] for colormap application
            img_normalized = (sliced_img_hor - np.min(sliced_img_hor)) / (np.ptp(sliced_img_hor) + 1e-8)

            # Apply the viridis colormap
            viridis_colormap = cm.get_cmap('viridis')
            img_colored = viridis_colormap(img_normalized)  # Result is an RGBA image

            # Convert RGBA to RGB (discard the alpha channel)
            img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

            # Convert RGB image to QImage
            height, width, channels = img_rgb.shape
            bytes_per_line = channels * width
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(q_img)

            # Resize the pixmap to fit the label, while preserving the aspect ratio
            max_width = 450  # Example max width
            max_height = 650  # Example max height
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)

            # Display the resized pixmap in the QLabel
            self.image_label.setPixmap(pixmap)

        else:
            # Handle invalid image format
            QMessageBox.critical(self, "Error", "Invalid image format for display.")

    def bg_subtraction_state_change(self, state):

        if self.loaded_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded to apply background subtraction.")
            self.safe_uncheck_checkbox()
            return

        if state == Qt.Checked:
          self.apply_background_subtraction()
          self.safe_uncheck_checkbox()

        elif state == Qt.Unchecked:
            self.loaded_image = self.original_image.copy()  # Restore the original image
            self.display_image(self.loaded_image)

    def apply_background_subtraction(self):
        self.background_images = [
            np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_0.npy'),
            np.load('D:/project/Data/save/background_x=-1.78mA_z=-2.55kV_1.npy'),
            np.load('D:/project/Data/save/background_x=-1.79mA_z=-2.55kV_2.npy'),
            np.load('D:/project/Data/save/background_x=-1.79mA_z=-2.55kV_3.npy'),
            np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_4.npy'),
            np.load('D:/project/Data/save/background_x=-1.75mA_z=-2.54kV_5.npy')]

        bck_avrg = np.mean(self.background_images, axis=0)

        if bck_avrg.shape != self.loaded_image.shape:
            raise ValueError("Image and background shapes do not match.")

        # Perform background subtraction
        self.loaded_image = self.loaded_image - bck_avrg
        self.loaded_image = np.clip(self.loaded_image, 0, 255).astype(np.uint8)

        # Display the modified image
        self.display_image(self.loaded_image)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Beam Image", "",
                                              "Image Files (*.png *.jpg *.bmp *.npy *.pgm)")
        if path:
            self.file_path = path
            try:
                if path.endswith('.npy'):
                    self.loaded_image = np.load(path)
                else:
                    self.loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if self.loaded_image is None:
                    raise ValueError("Failed to load image. File may be corrupted.")
                if len(self.loaded_image.shape) == 2:
                    self.is_grayscale = True
                elif len(self.loaded_image.shape) == 3:
                    self.is_grayscale = False
                else:
                    raise ValueError('Unexpected image format')

                self.original_image = self.loaded_image.copy()  # Save original
                self.display_image(self.loaded_image)

                #Apply background subtraction
                if self.bg_subtraction_checkbox.isChecked():
                    self.apply_background_subtraction()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def plot_beam_profile(self):
        try:
            # Ensure an image is loaded
            if not hasattr(self, 'loaded_image'):
                QMessageBox.critical(self, "Error", "No image loaded. Please load an image first.")
                return

            # Get user-defined parameters
            window_size = self.window_size_box.value()
            angle_rotation = self.angle_rotation_box.value()
            peak_distance = self.peak_distance_box.value()

            # Clear existing plots
            self.ax_hor.clear()
            self.ax_vert.clear()

            if self.file_path.endswith('.npy'):
                # Handle .npy file
                sliced_img_hor, params_hor, params_vert = plot_beam_profiles_npy(
                    self.loaded_image, self.ax_hor, self.ax_vert, window_size, angle_rotation
                )
            else:
                sliced_img_hor, params_hor, params_vert = plot_beam_profiles(
                        self.loaded_image, self.ax_hor, self.ax_vert, window_size
                    )
            # FWHM profile plotting
            if self.fwhm_checkbox.isChecked():
                self.ax_hor.clear()
                self.ax_vert.clear()

                profile_data = plot_fwhm_profile(self.loaded_image, self.ax_hor, self.ax_vert, window_size)
            else:
                self.display_image(self.loaded_image)

            # Check if the Gaussian filter checkbox is checked and apply the filter if needed
            if self.guassian_filter_checkbox.isChecked():
                self.ax_hor.clear()
                self.ax_vert.clear()
                # Re-plot the beam profiles with the filtered image
                sliced_img_hor, params_hor, params_vert = plot_smoothed_beam_profiles(
                    self.loaded_image, self.ax_hor, self.ax_vert, window_size, angle_rotation
                )
            else:
                self.display_image(self.loaded_image)

            if self.electron_train_checkbox.isChecked():
                self.ax_hor.clear()
                self.ax_vert.clear()
                sliced_img_hor, pop_all, profile_sliced_vert, params_vert = electron_beam_train_plots(
                    self.loaded_image, self.ax_hor, self.ax_vert, window_size, peak_distance
                )
            else:
                self.display_image(self.loaded_image)

            # Display the sliced image
            if sliced_img_hor is not None:
                self.display_image(sliced_img_hor)

            # Update Gaussian parameters
            _, mean_hor, sigma_hor, _ = params_hor
            fwhm_hor = 2.355 * params_hor[2]
            _, mean_vert, sigma_vert, _ = params_vert
            fwhm_vert = 2.355 * params_vert[2]

            self.mean_hor_value.setText(f"{mean_hor:.2f}")
            self.sigma_hor_value.setText(f"{sigma_hor:.2f}")
            self.fwhm_hor_value.setText(f"{fwhm_hor:.2f}")
            self.mean_vert_value.setText(f"{mean_vert:.2f}")
            self.sigma_vert_value.setText(f"{sigma_vert:.2f}")
            self.fwhm_vert_value.setText(f"{fwhm_vert:.2f}")

            # Invert y-axis for vertical plot
            self.ax_vert.invert_yaxis()
            self.canvas_hor.draw()
            self.canvas_vert.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during profile plotting: {str(e)}")

        self.ax_vert.invert_yaxis()
        self.canvas_hor.draw()
        self.canvas_vert.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamGUI()
    window.show()
    sys.exit(app.exec_())