# Parking Spot Detection

This project detects parking spot occupancy using deep learning and computer vision techniques. It includes training and inference scripts for YOLO and CNN models, as well as a Flask web server for visualization and alerting.

## Features
- Parking spot detection using YOLOv8 and custom CNN models
- Data collection and augmentation scripts
- Real-time inference and visualization
- Telegram alert integration for wrong parking
- Jupyter notebooks for training and experimentation

## Folder Structure
- `main.py` - Flask web server and main application
- `send_alert.py` - Telegram alert integration
- `datacollection.ipynb` - Data collection notebook
- `model_train/cnn/` - CNN training notebooks
- `model_train/yolo/` - YOLO training and inference notebooks
- `cropped_img/` - Region of interest images
- `templates/` - HTML templates for web UI

## Installation
1. Clone the repository
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Usage
- Run the Flask server:
  ```cmd
  python main.py
  ```
- Train YOLO model:
  Open `model_train/yolo/main.ipynb` in Jupyter and run the training cells.
- Train CNN model:
  Open `model_train/cnn/parking_spot_train_3_classes.ipynb` in Jupyter and run the training cells.

## Requirements
See `requirements.txt` for all required Python packages.

## License
MIT License
