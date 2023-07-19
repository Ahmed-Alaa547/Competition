# Library for numerical computations and array operations
import numpy as np           
# Library for data visualization  
import matplotlib.pyplot as plt 
# Library for interacting with the operating system  
import os                         
# Library for image processing (OpenCV) 
import cv2      
# Library for creating progress bars                   
from tqdm import tqdm             
# Function for calculating image similarity
from skimage.metrics import structural_similarity as ssim   
# High-level API for building and training deep learning models
from tensorflow import keras         
# Main folder 
DATADIR = 'Petimages'
CATEGORIES = ["Normal", "COVID", "Tuberculosis", "Viral Pneumonia", "Bacterial Pneumonia", "brain tumors YES", "brain tumors NO"]

IMG_SIZE = 100

training_data = []

def create_training_data():
    # Populate the training_data list with training examples
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# Shuffle the training data
np.random.shuffle(training_data)

# Split the data into features (X) and labels (y)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Normalize the pixel values
X = X / 255.0
'''
# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=32, epochs=20)

# Save the trained model
model.save('model.h5')
'''
# Load the trained model
model = keras.models.load_model('model.h5')

# Function to preprocess and test an image
def test_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    normalized_img = new_array / 255.0  # Normalize the image data
    reshaped_img = np.reshape(normalized_img, (1, IMG_SIZE, IMG_SIZE, 1))  # Reshape for model input
    prediction = model.predict(reshaped_img)
    class_index = np.argmax(prediction)
    predicted_class = CATEGORIES[class_index]
    return predicted_class, new_array

# Prompting the user for input folder
folder_path = input("Enter the path of the folder containing the images: ")

# Process each image in the folder
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    
    # Check if the file is an image
    if not os.path.isfile(image_path):
        continue
    
    # Check if the file is a valid image format
    valid_extensions = ['.png', '.jpg', '.jpeg']
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in valid_extensions:
        continue
    
    # Process the image
    predicted_class, test_array = test_image(image_path)
    print(f"Image: {filename}  |  Predicted Class: {predicted_class}")

    # Calculate similarity between input image and training images
    similarity_scores = []
    class_names = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            training_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_training_img = cv2.resize(training_img, (IMG_SIZE, IMG_SIZE))
            similarity = ssim(test_array, resized_training_img)
            similarity_scores.append(similarity)
            class_names.append(category)

    # Calculate average similarity
    average_similarity = np.mean(similarity_scores)
    class_labels = ["Normal", "COVID", "Tuberculosis", "Viral Pneumonia", "Bacterial Pneumonia", "brain tumors YES", "brain tumors NO"]
    similarity_per_class = [round(np.mean(similarity_scores[i:i+10]), 3) for i in range(0, len(similarity_scores), 10)]
    similarity_info = list(zip(similarity_per_class, class_labels))
    print("The average similarity between the input image and training images is:", similarity_info)
    print("The predicted class is:", predicted_class)

    # Display the image with the predicted class label
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array, cmap='gray')
    plt.title("Predicted Class: " + predicted_class)
    plt.axis("off")
    plt.show()
