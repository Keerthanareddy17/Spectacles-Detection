#  Glasses Detection with Lightweight Deep Learning (TFLite Ready)

A lightweight, high-accuracy deep learning model designed to detect whether a person is wearing **glasses or not**, even when the face is tilted up to **75°** and under some edge cases such as low light environments, blurred images etc. 

---

## 🎯 Objective

To build a **compact yet accurate glasses detection model** using deep learning that:

- Achieves **≥ 85% accuracy**
- Works on **angled faces** (up to 75°)
- Is optimized for deployment with **TensorFlow Lite**
- Has a model size of **≤ 30MB**

---

## 🚀 Approach

### 1. Dataset
- **CelebA** dataset with 200K+ celebrity images
- Binary labels: `Eyeglasses` → {1: Glasses, 0: No Glasses}
- Used official train/val/test splits

### 2. Preprocessing
- Label conversion and metadata creation
- Images resized to 128x128
- Normalized pixel values to [0,1]

### 3. Class Imbalance
- Computed and used `class_weight` during training to handle imbalance between "glasses" and "no glasses" classes

### 4. Model Architecture
- **Base**: MobileNetV2 (ImageNet pretrained)
- **Custom head**: GAP → Dense(128, ReLU) → BatchNorm → Dropout(0.4) → Dense(1, Sigmoid)

### 5. Augmentation
- Custom augmentations for angle robustness:
  - Rotation (±45°), brightness/contrast shifts
  - Horizontal flips, random crops, Gaussian noise

### 6. Training Strategy
- **Phase 1**: Train custom head with base frozen
- **Phase 2**: Fine-tune base model with lower learning rate
- Early stopping and LR scheduling used for stability

### 7. Evaluation
- Evaluated on held-out test set and individual images
- Consistent performance across batch and real-world inferences

<img width="704" alt="Screenshot 2025-04-29 at 3 13 12 PM" src="https://github.com/user-attachments/assets/0127d866-efca-4dfc-a13e-e6993b0196c5" />


<img width="1278" alt="Screenshot 2025-04-29 at 3 40 13 PM" src="https://github.com/user-attachments/assets/aac00a5f-88a8-4290-bdcb-16f5223dff6a" />

---
## 📂 Project Structure

- preprocessing.ipynb : Data preprocessing and metadata generation
- train.py : Model training with augmentation and class weights
- model.py : The model architecture
- compute_class_weights.py : Script to calculate the class weights
- convert_to_tflite.py : Converts the keras saved model to tflite format
- saved_models/ : Folder contaning the best models saved!
- requirements.txt : All the necessary dependencies
- test_model.py : This script is for evaluating the model on a single image before converting it into tflite format
- test_tflite_single_image.py : This script is for testing the model on a single image after converting it into tflite format
- test_tflite.py : This script is for testing the model on the test data after converting to tflite format
---
## 📌 If you want to run the project from scratch -

### 1. Clone this repo

```bash
git clone https://github.com/Keerthanareddy17/Spectacles-Detection.git
cd Spectacles-Detection
```

### 2. Create a Virtual Environment

Run the following command to create a new virtual environment in the project directory:

```bash
python -m venv venv
```
Activate the Virtual Environment 
- On Windows:
  ```
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
```
source venv/bin/activate
```
### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies from requirements.txt:

```
pip install -r requiremnts.txt
```
### 4. Preprocess the dataset
- Download the CelebA dataset and extract it from Kaggle [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset]
- Update paths in preprocessing.ipynb
- Run the notebook to generate metadata and verify images

### 5. Train the model

```
python train.py
```
### 6. Convert to TensorFlow Lite

```
python convert_to_tflite.py
```

### 7. Evaluate
Run the various scripts available to test out the final .tflite model!
