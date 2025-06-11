# Nighttime Surveillance System

This repository contains various scripts for video capture, face detection,
image enhancement and remote prediction using DeepFace.

## Structure

- `client/` – Client-side utilities for capturing video frames,
  controlling GPIO sensors and sending images to the server.
- `server/` – Flask server that receives uploaded images and performs face
  recognition.
- `predict_app.py` – Experimental siamese network for similarity prediction.
- `enhancement.py` and `lowlight.py` – Scripts to preprocess images used for
  training or evaluation.

## Usage

These scripts are provided as prototypes and may require hardware such as a
Raspberry Pi camera or IR sensor. Install dependencies from the respective
`requirements.txt` files before running a module.
