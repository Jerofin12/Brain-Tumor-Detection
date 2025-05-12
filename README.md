# 🧠 Gradio Brain Tumor Classification Gradio
This repository is a deep learning-based Brain Tumor Classification System that uses a Convolutional Neural Network (CNN) to classify MRI brain scans into one of four categories:

1.Glioma
2.Meningioma
3.No Tumor
4.Pituitary Tumor

It also provides a Gradio web interface for easy image uploads and predictions.

# 🚀 Features
1.Image classification using CNN
2.Four-class tumor detection
3.Interactive Gradio interface
4.Dataset loaded from Google Drive

# 📁 Dataset
The dataset is structured as follows inside Google Drive:

/Brain Tumor Segmentation/Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
Each folder contains MRI images belonging to the corresponding tumor class.

# Input Image
GitHubLearnOpenCV+4GitHub+4GitHub+4

# 🧰 Dependencies
Install the required Python libraries:

pip install gradio opencv-python-headless numpy tensorflow scikit-learn

For Google Colab:

from google.colab import drive
drive.mount('/content/drive')

# 🧠 Model Architecture
A CNN model built using TensorFlow/Keras with the following layers:

1.3× Conv2D + MaxPooling + BatchNorm
2.GlobalAveragePooling
3.Dense Layer (ReLU)
4.Dropout
5.Output Layer (Softmax)

# 🏗️ How It Works
1.Loads and preprocesses images from the Google Drive dataset.
2.Trains a CNN model to classify into 4 categories.
3.Provides a Gradio interface to upload and classify a new brain scan image.

# 🖼️ Using the Gradio Interface
Once you run the script:
interface.launch()
A Gradio interface will appear. You can:
Upload an MRI image
Get the predicted tumor type

# 🧪 Example Prediction
Upload a sample image like below:

sample_image.jpg
You’ll get a prediction like:

Prediction: Glioma (92% confidence)

# 📊 Model Training
The model is trained with:

Image size: 128×128
Batch size: 16
Epochs: 10
Loss: Categorical Crossentropy
Optimizer: Adam

# 📎 File Structure

├── gradio_brain_tumor_classification.py   # Main Python script
├── README.md                   # Project README
# 🔒 License
This project is open-source under the MIT License.

# ✅ Conclusion
This project demonstrates the power of deep learning in the field of medical image analysis. By leveraging a Convolutional Neural Network (CNN), we can effectively classify brain MRI scans into four tumor categories with high accuracy. The use of Gradio enhances usability by providing a simple, interactive interface for real-time predictions. With further improvements, such as training on a larger dataset or optimizing the architecture, this system has the potential to assist medical professionals in early tumor detection and diagnosis.

