# Speech Commands Classification with TensorFlow

This project trains a convolutional neural network (CNN) to classify speech commands using the Speech Commands dataset.

## Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/akotu235/speech-commands-classifier.git
cd speech-commands-classifier
```

## Python Version

Ensure Python 3.10.11 is installed before proceeding. You can verify your Python version with:

```bash
python --version
```

## Setup Virtual Environment
It is recommended to use a virtual environment for managing dependencies. Follow these steps:

1. Create a virtual environment:

    - On Linux/macOS:
      ```bash
      python3 -m venv venv
      ```
    - On Windows:
      ```bash
      python -m venv venv
      ```

2. Activate the virtual environment:

    - On Linux/macOS:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. Run the Python script:

   ```bash
   python speech_commands_classifier.py
   ```

2. The script will:
   - Load the Speech Commands dataset.
   - Preprocess the audio data.
   - Train a Convolutional Neural Network (CNN) model.
   - Evaluate the model on the test dataset.