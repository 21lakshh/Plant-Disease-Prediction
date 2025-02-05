# Kisaan Saathi - Crop Disease Detection & AgriAssist Support

## Overview
Kisaan Saathi is a comprehensive solution aimed at empowering farmers with advanced crop disease detection and personalized assistance. The project is divided into two main components:

1. **Crop Disease Detection Model**
2. **Streamlit Web Application with AgriAssist Chatbot**

---

## Crop Disease Detection Model

### Dataset
- **Dataset Used**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- The dataset contains images of crops categorized into 38 different disease classes.

### Model Details
- **Architecture**: Pretrained Xception model
- **Training Approach**:
  - Used `ImageDataGenerator` for augmenting the training dataset.
  - Validation and test datasets were not augmented.
- **Optimization**:
  - Initially trained the model with frozen layers.
  - Improved accuracy by unfreezing layers for fine-tuning.

### Metrics and Performance
- **Initial Metrics**:
  - **Training Accuracy**: 98.78%
  - **Validation Accuracy**: 99.14%
  - **Test Accuracy**: 99.05%
  - Generated a confusion matrix.
  - Analyzed metrics: Precision, Recall, F1-Score, and Support.
- **Final Accuracy**:
  - **Training Accuracy**: 99.81%
  - **Validation Accuracy**: 99.69%
  - **Test Accuracy**: 99.72%
- **Final Metrics**:
  - Confusion matrix and analysis of Precision, Recall, F1-Score, and Support were performed again after fine-tuning.

---

## Streamlit Web Application with AgriAssist Chatbot

### Features
1. **Integration with the Crop Disease Detection Model**:
   - Users can upload crop images to detect diseases.
   - Displays the predicted disease along with confidence scores.

2. **AgriAssist Chatbot**:
   - Powered by the Gemini-Pro model through the Gemini API.
   - Multilingual support for better farmer accessibility:
     - **Languages Supported**: Hindi, English, Kannada, Punjabi
   - Provides detailed insights on:
     - Cause of the detected disease  
     - Prevention measures  
     - Treatment solutions  


## Project Structure
```
App/
│── main.py               # Streamlit-web Application
│── class_indices.json      # All possible classes that can be predicted 
│── config.json     # contains the Gemeni API Key  
│── trained_model/
│   ├── plant_disease_prediciton.h5      # model file
│── requirements.txt     # List of Python dependencies
```
Please download the plant_disease_prediction.h5 file through this drive link: 
```sh
https://drive.google.com/file/d/1PqrcW3zoCfQlyKB58uXOBpxUKvtO0rSh/view?usp=sharing
```
## Screenshots
![image1](Images/image1.png)
![image2](Images/image2.png)
![image3](Images/image3.png)

### How It Works
1. **Upload an Image**: Farmers or users can upload a crop image to the web application.
2. **Disease Detection**: The application classifies the disease using the pretrained Xception model.
3. **AgriAssist Assistance**:
   - Choose between the predefined questions or use a custom prompt in any one of the listed languages.  
---

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/21lakshh/Kisaan-Saathi.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```
---

## Contributing
We welcome contributions! Feel free to open issues or submit pull requests.

---

## Acknowledgments
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) for providing the dataset.
- Gemini API for enabling advanced AI functionalities.
