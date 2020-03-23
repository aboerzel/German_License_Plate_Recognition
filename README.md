# Udacity Capstone Project
# Machine Learning Nanodegree 2020
 
### Topic : Android App for German License Plate Recognition

This project is to create a small Android app that allows to recognize the license plate of a car quickly and easily with the camera of a smartphone or tablet and translate the license into plain text. 
The app marks the recognized license plate within the camera image with a bounding box and displays the determined license in plain text as annotation above or below the bounding box, as can be seen in the following example:

![](documentation/demo_video.gif)

## Prerequisites
**For training the models of this project it is strongly recommended to use a computer with GPU support!**

## Setup Instructions
First create an change to a project directory and clone the project using the following command:
```
https://github.com/aboerzel/German_License_Plate_Recognition.git
```
This will download the repo to the current project directory.

For this project we need 2 development environments, a Tensorflow 1.13 environment for license plate detection with the Tensorflow Object Detection API, and a Tensorflow 2.0 environment for license recognition.

**Important:** The Tensorflow Object Detection API is currently **not compatible** with Tensorflow 2.0!

#### Setup Tensorflow 1.13 and Object Detection API
Change to the `tf_object_detection` folder and create a Tensorflow 1.13 environment `tf1.13` for the Tensorflow Object Detction API and activate it
```
cd ./tf_object_detection
conda env create -f tf1.13.yml
activate tf1.13
```
Compile Protobufs
```
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
```
Install the `object_detection` python package
```
python Setup.py build
python Setup.py install
```
Install COCO API
```
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
```

#### Setup Tensorflow 2.0
Deactivate the current environment, change back to the project root folder and create a Tensorflow 2.0 environment `tf2.0`
```
deactivate
cd ..
conda env create -f tf2.0.yml
```

#### Install Android Studio
Download and install [Android Studio](https://developer.android.com/studio)

## Documentation

Here you can find the to the project proposal and the write-up of the final project:
- The [project proposal](proposal/proposal.pdf) 
- The final [project report](documentation/report.pdf)

To train the plate detection model activate the `tf1.13` environment and run the following notebooks from the project root folder: 
- [Data Exploration and Preparation](1_License_Plate_Detection_Data_Exploration_And_Preparation.ipynb)
- [Plate Detection Model Training and Evaluation](2_License_Plate_Detection_Model_Training_And_Evaluation.ipynb)

To train the license recognition model activate the `tf2.0` environment and run the following notebooks from the project root folder:
- [Data Collection and Exploration](3_License_Recognition_Data_Collection_And_Exploration.ipynb)
- [License Recognition Model Training and Evaluation](4_License_Recognition_Model_Training_And_Evaluation.ipynb)

The following notebook demonstrates the complete workflow by first extracting the license plate from the camera image using the Plate Detector and then determining the license text using the License Recognizer: 
- [License Detection And Recognition Workflow](5_License_Recognition_Workflow.ipynb)

## Android App
The Android App (APK file) can be downloaded from [here](https://drive.google.com/file/d/1gJZhZE3F3gq35Wn_J9AUCSiN4sP9pIqh/view?usp=sharing).
 