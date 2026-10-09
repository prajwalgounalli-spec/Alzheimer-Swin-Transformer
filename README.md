# An MRI-Based Swin Transformer Approach for Automated Alzheimer's Disease Recognition

## 1. Project Overview

This project implements an MRI image classification system using a pretrained Swin Transformer-Tiny model. The system classifies uploaded brain MRI images into two dataset categories:

- **AD:** Alzheimer's disease dataset class
- **NC:** Non-demented dataset class

A Streamlit web dashboard allows users to upload an MRI image and view the model's predicted class and class scores.

**Important:** This is an academic research prototype, not a medical diagnostic system.

## 2. Objectives

- Prepare and organize MRI images for binary classification.
- Use transfer learning with Swin Transformer-Tiny.
- Fine-tune the pretrained model for AD and NC classification.
- Evaluate the model using validation and test data.
- Develop a simple interactive Streamlit dashboard.
- Display predictions and class score distributions.

## 3. Model Architecture

**Model:** Swin Transformer-Tiny

**Approach:** Transfer learning and fine-tuning

**Input resolution:** 224 × 224 pixels

**Input channels:** 3 (RGB)

**Output classes:** AD and NC

The model uses hierarchical shifted-window self-attention to learn image features. A classification head produces scores for the two target classes.

## 4. Dataset

The project uses a publicly available Alzheimer's MRI image dataset.

The selected dataset is organized into training, validation, and testing subsets. The local test split contains:

| Class | Number of JPG images |
|---|---:|
| AD | 66 |
| NC | 67 |
| Total | 133 |

Dataset labels describe the source image categories. They do not establish a clinical diagnosis.

The dataset contains augmented images and does not provide patient identifiers. Therefore, the split should not be described as patient-independent, and possible dataset-level leakage remains a limitation.

## 5. Data Preprocessing

The dashboard applies the following transformations:

1. Convert the uploaded image to RGB.
2. Resize it to 224 × 224 pixels.
3. Convert the image to a PyTorch tensor.
4. Normalize using ImageNet mean and standard deviation.
5. Pass the image through the trained Swin Transformer-Tiny model.

The inference preprocessing should match the preprocessing used during training.

## 6. Reported Model Results

The selected Experiment 5 checkpoint achieved the following previously recorded results:

| Metric | Result |
|---|---:|
| Best validation accuracy | 82.84% |
| Test accuracy | 75.19% |

These results are specific to the recorded experiment and dataset split. They are not guaranteed to reproduce on other data.

The test set has been used to compare multiple experiments, so its results should be considered exploratory rather than an untouched final evaluation.

## 7. Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab
- Visual Studio Code

## 8. Project Structure

```text
Alzheimer_Swin_ADNC_Project/
├── app.py
├── models/
│   ├── swin_model.py
│   ├── test_swin.py
│   └── best_swin_tiny_v5.pth
├── preprocessing/
│   ├── dataloader.py
│   └── transforms.py
├── dataset_group_split/
│   ├── train/
│   │   ├── AD/
│   │   └── NC/
│   ├── val/
│   │   ├── AD/
│   │   └── NC/
│   └── test/
│       ├── AD/
│       └── NC/
├── results/
└── README.md
```

The project may contain additional files depending on the local setup.

## 9. Installation

Python 3.10 or a compatible version is recommended.

Open a terminal in the project directory and activate the existing virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages if they are not already installed:

```powershell
python -m pip install torch torchvision streamlit pillow
```

Use compatible PyTorch and TorchVision versions for your environment.

## 10. Run the Dashboard

From the project root, execute:

```powershell
python -m streamlit run app.py
```

Open the local address shown in the terminal, normally:

```text
http://localhost:8501
```

### Dashboard Usage

1. Open the Streamlit dashboard.
2. Upload a JPG, JPEG, or PNG MRI image.
3. Click **Run MRI Classification**.
4. Review the predicted dataset class and the displayed AD and NC scores.

Keep the terminal running while using the dashboard.

## 11. Hardware

The development environment used an Intel Core i5 laptop with 16 GB RAM. Model training was performed using Google Colab with a Tesla T4 GPU. Local CPU inference is supported by the current dashboard.

## 12. Limitations

- The dataset is limited and may not represent real-world clinical MRI data.
- Patient identifiers are unavailable.
- Augmented images can introduce bias or leakage.
- Model scores are not calibrated clinical probabilities.
- The system has not been validated for medical use.
- Predictions must not be used for diagnosis or treatment decisions.

## 13. Future Enhancements

- Evaluate on an independent dataset with patient-level separation.
- Improve model generalization and class-wise performance.
- Add confusion matrix, precision, recall, F1-score, and ROC curves.
- Investigate explainability methods such as Grad-CAM-compatible approaches where technically appropriate.
- Improve the dashboard with a model evaluation report.

## 14. Conclusion

This project demonstrates an MRI image classification workflow using Swin Transformer-Tiny and a Streamlit interface. It combines image preprocessing, transfer learning, model inference, and interactive visualization in an academic research prototype.

---

**Project type:** M.Tech Research Project  
**Model:** Swin Transformer-Tiny  
**Task:** Binary MRI image classification (AD vs NC)

## Model Performance

| Evaluation Metric | Result |
|---|---:|

Detailed results: 
esults/model_evaluation_summary.txt
Per-image predictions: 
esults/test_predictions.csv


