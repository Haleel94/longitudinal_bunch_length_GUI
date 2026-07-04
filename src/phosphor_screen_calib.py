import sys
import numpy as np
import cv2
from scipy.optimize import least_squares
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QWidget, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

class BeamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Profile Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # Variables to store circle parameters
        self.xc = None
        self.yc = None
        self.r = None

        # Matplotlib figure for displaying image and circle
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Buttons
        self.load_button = QPushButton("Load Image")
        self.detect_button = QPushButton("Detect Circle")
        self.plot_button = QPushButton("Fit Points and Plot Circle")
        self.detect_button.setEnabled(False)

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.plot_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect buttons
        self.load_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.detect_circle)
        self.plot_button.clicked.connect(self.detect_circle1)

        # Connect Matplotlib mouse motion event
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Image data
        self.loaded_image = None

    def load_image(self):
        """Load an image from disk."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Beam Image", "",
                                              "Image Files (*.png *.jpg *.bmp *.npy *.pgm)")
        if path:
            try:
                if path.endswith('.npy'):
                    self.loaded_image = np.load(path)
                else:
                    self.loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if self.loaded_image is not None:
                    self.ax.clear()
                    self.ax.imshow(self.loaded_image, cmap="gray")
                    self.canvas.draw()
                    self.detect_button.setEnabled(True)
                else:
                    QMessageBox.critical(self, "Error", "Failed to load the image. Please check the file format.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the image: {str(e)}")

    def detect_circle(self):
        """Detect edges and fit a circle to the image."""
        if self.loaded_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        try:
            # Preprocess the image: blur and normalize for better edges
            blurred = cv2.GaussianBlur(self.loaded_image, (5, 5), 0)  # Reduce noise
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)  # Enhance contrast

            # Adaptive Canny edge detection
            median_val = np.median(normalized)
            lower_thresh = int(max(0, 0.66 * median_val))  # Lower threshold (66% of median)
            upper_thresh = int(min(255, 1.33 * median_val))  # Upper threshold (133% of median)
            edges = cv2.Canny(normalized, lower_thresh, upper_thresh, apertureSize=3, L2gradient=True)

            # Extract coordinates of detected edges
            y_coords, x_coords = np.where(edges > 0)  # >0 since edges are binary (0 or 255)

            # Fit circle using weighted least squares based on pixel intensity
            weights = self.loaded_image[y_coords, x_coords] / 255.0  # Weight by pixel intensity (0 to 1)
            self.xc, self.yc, self.r = self.fit_circle(x_coords, y_coords, weights)

            # Display the image and fitted circle
            self.ax.clear()
            self.ax.imshow(self.loaded_image, cmap="gray")
            circle_patch = Circle((self.xc, self.yc), self.r, color="r", fill=False, linewidth=3, linestyle='--')
            self.ax.add_patch(circle_patch)
            self.ax.set_title(f"Circle Fitted: Center=({self.xc:.2f}, {self.yc:.2f}), Radius={self.r:.2f}")
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during circle detection: {str(e)}")

    def detect_circle1(self):
        """Detect and fit a circle using predefined coordinates."""
        try:
            # Predefined coordinates for circle fitting
            yx_coords = [(45, 283), (103, 715), (1207, 374), (1078, 799),
                         (1173, 637), (1078, 54), (1206, 502), (142, 73), (1141, 707)]

            # Separate the coordinates into x and y
            x_coords, y_coords = zip(*yx_coords)

            # Fit the circle to the coordinates
            self.xc, self.yc, self.r = self.fit_circle(np.array(x_coords), np.array(y_coords))

            # Clear the axis and display the image
            self.ax.clear()
            self.ax.imshow(self.loaded_image, cmap="gray")

            # Add the fitted circle
            circle_patch = Circle((self.xc, self.yc))
            self.ax.add_patch(circle_patch)

            # Highlight the provided coordinates
            self.ax.scatter(x_coords, y_coords)

            # Update the title with circle parameters
            self.ax.set_title(f"Fitted Circle: Center=({self.xc:.2f}, {self.yc:.2f}), Radius={self.r:.2f}")
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during circle detection: {str(e)}")

    def on_mouse_move(self, event):
        """Handle mouse motion events."""
        if event.inaxes == self.ax and self.xc is not None and self.yc is not None and self.r is not None:
            # Check if the cursor is near the circle's circumference
            distance = np.sqrt((event.xdata - self.xc) ** 2 + (event.ydata - self.yc) ** 2)
            if abs(distance - self.r) < 5:  # Threshold for "near the circle"
                # Display circle parameters dynamically
                self.ax.set_title(f"Center: ({self.xc:.2f}, {self.yc:.2f}), Radius: {self.r:.2f}")
            else:
                self.ax.set_title("Circle Fitted")
            self.canvas.draw()

    @staticmethod
    def circle_model(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    @staticmethod
   # def fit_circle(x, y):
        #x_m = np.mean(x)
        #y_m = np.mean(y)
        #r_m = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))
        #initial_guess = [x_m, y_m, r_m]
        #result = least_squares(BeamGUI.circle_model, initial_guess, args=(x, y))
        #return result.x  # xc, yc, r

    @staticmethod
    def fit_circle(x, y, weights=None):
        """Fit a circle using weighted least squares."""
        x_m = np.average(x, weights=weights) if weights is not None else np.mean(x)
        y_m = np.average(y, weights=weights) if weights is not None else np.mean(y)
        r_m = np.average(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2), weights=weights) if weights is not None else np.mean(
            np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))

        initial_guess = [x_m, y_m, r_m]
        result = least_squares(BeamGUI.circle_model, initial_guess, args=(x, y), loss='soft_l1', f_scale=0.5)
        return result.x  # xc, yc, r


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamGUI()
    window.show()
    sys.exit(app.exec_())
