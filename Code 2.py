import os  
import cv2  
import numpy as np  
from tensorflow import keras  
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QScrollArea, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve  
from PyQt5.QtGui import QPixmap
import sqlite3  

CATEGORIES = ["Normal", "COVID", "Tuberculosis", "Viral Pneumonia", "Bacterial Pneumonia", "brain tumors YES", "brain tumors NO"]

class ResultWindow(QMainWindow):
    # Constructor with file_paths (list of image file paths) and img_size (desired image size) as parameters
    def __init__(self, file_paths, img_size):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Result")
        self.setGeometry(100, 100, 500, 500)
        self.setStyleSheet("background-color: #333333; color: white;")

        self.img_size = img_size
        self.model = keras.models.load_model('model.h5')  # Load the pre-trained model

        # Define fade-in and fade-out animations for the window
        # Animations are used to create a visual effect when the window is shown or hidden, and have  have a duration of 500 milliseconds
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutQuad)

        self.fade_out_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out_animation.setDuration(500)
        self.fade_out_animation.setStartValue(1)
        self.fade_out_animation.setEndValue(0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_out_animation.finished.connect(self.close)

        if file_paths:
            self.display_results(file_paths)

    # Method to display the results for each image
    def display_results(self, file_paths):
        vbox = QVBoxLayout()
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        vbox_layout = QVBoxLayout(content_widget)

        for file_path in file_paths:
            # Read, resize, and normalize the image
            img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (self.img_size, self.img_size))
            normalized_img = new_array / 255.0
            reshaped_img = np.reshape(normalized_img, (1, self.img_size, self.img_size, 1))

            # Make a prediction using the loaded model
            prediction = self.model.predict(reshaped_img)
            class_index = np.argmax(prediction)
            predicted_class = CATEGORIES[class_index]

            # Add QLabel to display the image
            image_label = QLabel(self)
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaledToHeight(300, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setFixedSize(300, 300)  # Set the desired size of the image label
            vbox_layout.addWidget(image_label, alignment=Qt.AlignCenter)

            # Add QLabel to display the predicted class
            result_label = QLabel(self)
            result_label.setText(f"Predicted Class: {predicted_class}")
            result_label.setAlignment(Qt.AlignCenter) # which centers the text horizontally.
            vbox_layout.addWidget(result_label, alignment=Qt.AlignTop)

        vbox_layout.addStretch()  # Add stretch before adding buttons

        hbox_layout = QHBoxLayout() # display it in the window.

        # Add exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setStyleSheet("background-color: #666666; color: white;")
        self.exit_button.clicked.connect(self.exit_program)
        self.exit_button.setFixedSize(135, 30)  # Set the desired button size
        hbox_layout.addWidget(self.exit_button, alignment=Qt.AlignCenter)

        # Add import again button
        self.import_again_button = QPushButton("Import Again", self)
        self.import_again_button.setStyleSheet("background-color: #666666; color: white;")
        self.import_again_button.clicked.connect(self.import_again)
        self.import_again_button.setFixedSize(135, 30)  # Set the desired button size
        hbox_layout.addWidget(self.import_again_button, alignment=Qt.AlignCenter)

        vbox_layout.addLayout(hbox_layout)  # Add the horizontal layout to the vertical layout

        scroll_area.setWidget(content_widget)
        scroll_area.setFixedHeight(500)  # Set the desired height of the scroll area
        vbox.addWidget(scroll_area)

        central_widget = QWidget(self)
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)

    # Method called when the window is shown , and start the fade-in and fade-out animations when 
    # the window is shown or hidden, creating a smooth visual transition effect.
    def showEvent(self, event):
        self.fade_in_animation.start()

    # Method called when the window is hidden , the same also as above
    def hideEvent(self, event):
        self.fade_out_animation.start()

    # Method to exit the program
    def exit_program(self):
        self.close()
        app.quit()

    # Method to import images again
    def import_again(self):
        self.close()
        main_window.show()

# Class for the Main Window GUI
class MainWindow(QMainWindow):
    IMG_SIZE = 100

    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Image Classification")
        self.setGeometry(100, 100, 400, 200)
        self.setStyleSheet("background-color: #333333; color: white;")

        # Add import photo button
        self.import_photo_button = QPushButton("Import Photo", self)
        self.import_photo_button.setGeometry(50, 80, 150, 40)
        self.import_photo_button.setStyleSheet("background-color: #666666; color: white;")
        self.import_photo_button.clicked.connect(self.import_photo)

        # Add import folder button
        self.import_folder_button = QPushButton("Import Folder", self)
        self.import_folder_button.setGeometry(210, 80, 150, 40)
        self.import_folder_button.setStyleSheet("background-color: #666666; color: white;")
        self.import_folder_button.clicked.connect(self.import_folder)

        # Define fade-in and fade-out animations for the window
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutQuad)

        self.fade_out_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out_animation.setDuration(500)
        self.fade_out_animation.setStartValue(1)
        self.fade_out_animation.setEndValue(0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_out_animation.finished.connect(self.close)

    # Method called when the window is shown
    def showEvent(self, event):
        self.fade_in_animation.start()

    # Method called when the window is hidden
    def hideEvent(self, event):
        self.fade_out_animation.start()

    # Method to import a single photo
    def import_photo(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Import Photo", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            file_paths = [file_path]
            img_size = 100  # Set the desired image size here
            self.result_window = ResultWindow(file_paths, img_size)
            self.hide()
            self.result_window.show()

    # Method to import a folder of photos
    def import_folder(self):
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Import Folder")
        if folder_path:
            file_names = os.listdir(folder_path)
            file_paths = [os.path.join(folder_path, file_name) for file_name in file_names if
                          os.path.isfile(os.path.join(folder_path, file_name))]
            if file_paths:
                img_size = 100
                self.result_window = ResultWindow(file_paths, img_size)
                self.hide()
                self.result_window.show()

# Class for the Login Window GUI
class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Login")
        self.setGeometry(100, 100, 400, 200)
        self.setStyleSheet("background-color: #333333; color: white;")

        # Add username label
        self.username_label = QLabel("Username:", self)
        self.username_label.move(50, 40)

        # Add username input field
        self.username_input = QLineEdit(self)
        self.username_input.setGeometry(150, 40, 200, 30)

        # Add password label
        self.password_label = QLabel("Password:", self)
        self.password_label.move(50, 80)

        # Add password input field
        self.password_input = QLineEdit(self)
        self.password_input.setGeometry(150, 80, 200, 30)
        self.password_input.setEchoMode(QLineEdit.Password)

        # Add login button
        self.login_button = QPushButton("Login", self)
        self.login_button.setGeometry(180, 125, 100, 30)
        self.login_button.clicked.connect(self.login)

        self.show()

    # Method called when the login button is clicked
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # Verify username and password against the database
        if self.authenticate(username, password):
            main_window.doctor_name = username  # Set the doctor's name in the main window
            self.close()
            main_window.show()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password")

    # Method to authenticate the user against the database
    def authenticate(self, username, password):
        # Connect to the database
        conn = sqlite3.connect("user_database.db")
        cursor = conn.cursor()

        # Check if the username and password match in the database
        cursor.execute("SELECT * FROM user_database WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        # Return True if a matching user is found, False otherwise
        return user is not None

# Create the QApplication instance
app = QApplication([])

# Create the Login Window instance
login_window = LoginWindow()

# Create the Main Window instance
main_window = MainWindow()

# Show the Login Window
login_window.show()

# Execute the application event loop
app.exec_()
