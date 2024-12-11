# Facial Emotion Recognition App

This is a Facial Emotion Recognition app built using Convolutional Neural Networks (CNN) in Python. The app detects emotions from facial expressions captured in images or live video feeds.

## Requirements

- Python 3.6+
- TensorFlow 2.x or Keras
- OpenCV
- numpy
- dlib
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository or download the code:

    ```bash
    git clone https://github.com/yourusername/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

2. Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This model is trained on the **FER-2013** dataset (Facial Expression Recognition), which contains images of people with various facial expressions. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/msambare/fer2013).

## Model Architecture

The model is a Convolutional Neural Network (CNN) built to classify images into one of the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The architecture consists of multiple convolutional layers, pooling layers, and fully connected layers, followed by a softmax layer to predict the class.

## How to Run the App

### 1. Train the Model (Optional)

If you want to train the model yourself, use the following command:

```bash
python main.py
```

This will train the CNN on the FER-2013 dataset and save the trained model to `emotion_model.h5`.

### 2. Run the Facial Emotion Recognition

To run the application and recognize emotions from a webcam feed or image:

```bash
python test.py
```

- For webcam input, this will display the live camera feed, detecting emotions in real time.

### 3. Using the Pre-trained Model

If you have already trained the model, or have a pre-trained model (e.g., `emotion_model.h5`), you can skip the training step and use the following command to recognize emotions:

```bash
python recognize_emotion.py
```

The model will use the pre-trained weights and detect emotions from the input.

## Code Files

- `main.py`: This script trains the CNN model on the FER-2013 dataset.
- `test.py`: This script runs the app to detect emotions from an image or live video feed.
- `emotion_model.h5`: The pre-trained model file (generated after training).

## Example Usage

1. **Webcam Emotion Recognition**: Launch the app and the webcam will open, detecting emotions.

    ```bash
    python test.py
    ```

2. **Image Emotion Recognition**: Recognize emotions in a specific image.


## Contributing

Feel free to fork this repository, raise issues, or contribute improvements! Contributions are welcome.
