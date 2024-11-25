# **Plant Disease Prediction**  
This is a Streamlit web application for predicting plant diseases using machine learning and computer vision techniques. The model analyzes plant leaf images and predicts the disease, helping farmers and gardeners take timely action.  

## **Overview**  
Plant diseases can severely impact agricultural productivity and food security. Early and accurate detection of plant diseases is crucial for effective management and to minimize losses. The project includes a Streamlit web application that allows users to upload images of plant leaves. The app processes these images using a trained deep learning model to predict whether the plant is healthy or affected by a specific disease.  

### **Features**  

- Upload Image: Users can upload an image of a plant leaf.  
- Disease Prediction: The app predicts the disease (or indicates a healthy plant).  
- Confidence Score: Provides a confidence score for the prediction.  
- Disease Information: Displays basic information and possible remedies for the predicted disease.  
- User-Friendly Interface: Simple and intuitive design powered by Streamlit.  

Dataset Features:
10849 images belonging to 38 classes.  
 0: 'Apple___Apple_scab'  
 1: 'Apple___Black_rot'  
 2: 'Apple___Cedar_apple_rust'  
 3: 'Apple___healthy'  
 4: 'Blueberry___healthy'  
 5: 'Cherry_(including_sour)___Powdery_mildew'  
 6: 'Cherry_(including_sour)___healthy'  
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'  
 8: 'Corn_(maize)___Common_rust_'  
 9: 'Corn_(maize)___Northern_Leaf_Blight'  
 10: 'Corn_(maize)___healthy'  
 11: 'Grape___Black_rot'  
 12: 'Grape___Esca_(Black_Measles)'  
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'  
 14: 'Grape___healthy'  
 15: 'Orange___Haunglongbing_(Citrus_greening)'  
 16: 'Peach___Bacterial_spot'  
 17: 'Peach___healthy'  
 18: 'Pepper,_bell___Bacterial_spot'  
 19: 'Pepper,_bell___healthy'  
 20: 'Potato___Early_blight'  
 21: 'Potato___Late_blight'  
 22: 'Potato___healthy'  
 23: 'Raspberry___healthy'  
 24: 'Soybean___healthy'  
 25: 'Squash___Powdery_mildew'  
 26: 'Strawberry___Leaf_scorch'  
 27: 'Strawberry___healthy'  
 28: 'Tomato___Bacterial_spot'  
 29: 'Tomato___Early_blight'  
 30: 'Tomato___Late_blight'  
 31: 'Tomato___Leaf_Mold'  
 32: 'Tomato___Septoria_leaf_spot'  
 33: 'Tomato___Spider_mites Two-spotted_spider_mite'  
 34: 'Tomato___Target_Spot'  
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'  
 36: 'Tomato___Tomato_mosaic_virus'  
 37: 'Tomato___healthy'  

# Data Pre-Processing:
- Images Resized to (224,224)  
- Converted to RGB
- Converted them to NumPy Arrays  
- ReScaled to similar values

# Model Architecture
model = Sequential()  

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  
model.add(MaxPooling2D(2, 2))  

model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D(2, 2))  


model.add(Flatten())  
model.add(Dense(256, activation='relu'))  
model.add(Dense(38, activation='softmax'))  

model.summary()  

# Model Evaluation:
- Accuracy on Training Data: 90.86%  
- Accuracy on Validation Data: 84.38%  

##### **Getting Started** 
Deployed using Streamlit as a web-application  
Clone the repo and install dependencies.  
Run the python file and try it yourself!  

you need to run the following commands in the terminal  
- **cd to Current working directory**  
- **streamlit run main.py**  
