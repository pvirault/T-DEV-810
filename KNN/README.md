# Pneumonia Classification with K-Nearest Neighbours (KNN)

### Libraries
The necessary libraries for image processing, data visualization, and machine learning tasks.

- Torch: Is an open source ML library used for creating deep neural networks and is written in the Lua scripting language 
```pip install torch```
- Torchvision: Is a package that consists of popular datasets, model architectures, and common image transformations for computer vision. 
```pip install torchvision```
- NumPy: A library for numerical computing in Python 
```pip install numpy```
- Seaborn: A library for data visualization
```pip install seaborn```
- Matplotlib: A library for data visualization
```pip install matplotlib```
- Scikit-learn: A machine learning library for Python
```pip install scikit-learn```

### Data Preprocessing
The **load_datasets** function load and edit images from the given directory. The following transformations are performed on the images:

- Convert it to greyscale as x-rays are black and white
- Resize the image to the VGG-16 input size of (224, 224)
- Convert the image to a tensor
- Normalize with std=0.5, mean=0.5
- Flatten the image

### KNN

The **KNN** function.
The objective is given a bunch of training data and pairs, find the k closest training examples and get their labels. Whatever label has the highest probability is what the test image is classified as.


![alt text](https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg)

Image Exemple : 
the green dot is our test xray image. Depending on how many neighbours we consider it will change the outcome of how we classify our test image. K = 2 will classify it as red (PNEUMONIA), while K = 5 will classify it as blue (NORMAL).

### Hyperparameter Grid Search
This performs grid search on some common K-values as well as some guesses to see what the best result ends up being.

### Model Evaluation
The trained model is used to predict the classes of the validation set, and the classification report, confusion matrix, and ROC curve are generated to evaluate the performance of the model.
