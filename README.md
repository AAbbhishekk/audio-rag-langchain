# Audio Query App

This repository contains a Streamlit app that allows users to upload an audio file (in WAV or MP3 format) and ask queries related to the audio content. The app utilizes various language models and libraries for audio processing, text embedding, and question-answering capabilities.

## Prerequisites

Before running the app, ensure that you have the required libraries installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Make sure to create a `.env` file with appropriate configurations.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open your browser and navigate to the provided URL (usually `http://localhost:8501`).

## Overview

The app includes the following components:

- **Audio File Upload**: Users can upload WAV or MP3 files containing audio content.

- **Audio Transcription**: The uploaded audio file is transcribed using the AssemblyAI Audio Transcription service.

- **Text Splitting**: The transcribed text is split into smaller chunks for further processing.

- **Text Embedding**: The text chunks are embedded using the Hugging Face Hub, specifically the "sentence-transformers/all-mpnet-base-v2" model.

- **Question-Answering (QA) Chain**: A QA chain is created using the ChatOpenAI model or a custom Hugging Face model ("HuggingFaceH4/zephyr-7b-beta"). The QA chain is responsible for answering user queries based on the embedded text chunks.

- **User Interface**: The Streamlit UI allows users to input queries and submit them for processing. The results, including the answer and relevant source documents, are displayed.

## Configuration

Ensure that you have a valid API key or configuration for the necessary services, and update the `.env` file accordingly.

## Notes

- The app uses the AssemblyAI Audio Transcription service, which may take several minutes to transcribe the uploaded audio file.

- The QA chain relies on the specified Hugging Face model, and you may customize the model parameters based on your requirements.

Feel free to explore and modify the code to suit your specific use case and integrate additional features as needed.
