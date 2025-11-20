*Emotion Detection*

A deep-learning–based facial emotion detection system that identifies human emotions from images or video input. The project uses a custom CNN model, dataset preprocessing, data augmentation, and facial recognition utilities to classify emotions such as happy, sad, angry, neutral, and more.





*Features*

- Custom CNN model for emotion classification

- Face detection module to extract faces from raw images

- Dataset loader and preprocessing pipeline

- Data augmentation for improving model performance

- Standalone emotion classifier script

- Modular codebase for easy extension





*Emotion-Detection*/
│
├── emotion_classifier/          # Pretrained classifier & utilities
│   └── emotionclassifier.py
│
├── emotion_det_files/           # Model weights, label files, etc.
│
├── src_files/                   # Core source code
│   ├── cnn.py                   # CNN model architecture
│   ├── dataset.py               # Dataset loader and preprocessing
│   ├── face_rec.py              # Face detection & extraction
│   ├── data_augmentation.py     # Augmentation pipeline
│   └── main.py                  # Training / testing entry point
│
└── README.md
