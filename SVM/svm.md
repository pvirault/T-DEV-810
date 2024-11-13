# Pneumonia Classification with SVM

### Libraries
The necessary libraries for image processing, data visualization, and machine learning tasks.

- NumPy: A library for numerical computing in Python
    ```pip install numpy```
- Matplotlib: A library for data visualization
    ```pip install matplotlib```
- OpenCV: A library for image processing and computer vision tasks
    ```pip install opencv-python```
- Keras: A high-level deep learning library that runs on top of TensorFlow
    ```pip install keras```
- Scikit-learn: A machine learning library for Python
    ```pip install scikit-learn```

### Data Preprocessing
The **get_data** function reads the images from the given directory, resizes them to the specified image size, and returns them as a numpy array. The data is then split into train, test, and validation sets, and then the data is normalized by dividing each pixel value by 255.

### Data Augmentation
The **ImageDataGenerator** function from the Keras library is used to augment the training data by applying various transformations to the images such as rotation, zoom, and flipping. The augmented data is then added to the original training set.

### Model Selection
The **GridSearchCV** function from the scikit-learn library is used to search for the best hyperparameters for the Support Vector Machine (SVM) model. The SVM model is then instantiated with the best parameters and trained on the entire training set.

### Model Evaluation
The trained model is used to predict the classes of the validation set, and the classification report, confusion matrix, and ROC curve are generated to evaluate the performance of the model.