# Intelligent Video Surveillance Using Deep learning System 🕵️‍♂️ 
IVSUDLS (Intelligent Video Surveillance Using Deep Learning System) aims to take real-time videos from CCTV as an input and pass it to the CNN model created with the help of transfer learning and detect ‘Shoplifting’, ‘Robbery’ , ’Break-In’,’Abuse’, ‘Arrest’ , ’Normal’ , ’Accident’,‘Arson’, ‘Explosion’ , ’Assault’,’Burglary’, ‘Fighting’ , ’Shooting’ or ’Stealing’  in the store and notify it to the owners as soon as it occurs. Finally the main motive is to provide a system that detects suspicious activities without human intervention and generates alert, thus making a huge revolution in today’s surveillance system.

<img src="https://github.com/OmRajpurkar/Alert-Generation-on-Detection-of-Suspicious-Activity-using-Transfer-Learning/blob/main/videoClassification/Shoplifting.gif" alt="alt text">

## Downloading and Configuring


* Download the [model](https://drive.google.com/file/d/1vtL4Isubeao4lm85V8YYsVsaUFiMFiEg/view?usp=sharing) and place it inside ‘project’ directory.

* Download the [dataset](https://drive.google.com/drive/folders/1x9IQo_wVObe6G0G0aH-hrPlv6Kk024jj?usp=sharing) and place it inside ‘project’ directory.

* Create a new Virtual Environment and Activate it.

* To save the dataset images create a directory named 'data' inside the 'project' directory. Inside the 'data' directory create 13 sub-directories each for ‘Shoplifting’, ‘Robbery’ , ’Break-In’,’Abuse’, ‘Arrest’ , ’Normal’ , ’Accident’,‘Arson’, ‘Explosion’ , ’Assault’,’Burglary’, ‘Fighting’ , ’Shooting’ or ’Stealing’. Store the training images for the respective classes inside those directories.

## Requirements

```bash
$ pip install -r requirements.txt
```

## Training the model

```bash
$ python train.py
```

## Predicting ‘Shoplifting’, ’Robbery’, ‘Break-In’ in Real-time using Webcam

```bash
$ python project.py 
```
