<div style="text-align:center; margin-bottom: 30px;">
    <h1 style="font-size: 34px; margin-bottom: 10px;">Emotion Detection</h1>
    <p style="font-size: 18px; color: #555; margin: 0;">
        A simple image-based emotion detection system built using a CNN model.
    </p>
</div>

<h2 style="margin-top:40px;">About the Project</h2>
<p style="line-height:1.6; font-size:16px;">
    This project focuses on detecting basic human emotions from facial images using a
    convolutional neural network. The setup includes preprocessing steps, a trained model,
    and separate scripts for face detection, dataset handling, and result prediction.
</p>

<h2 style="margin-top:30px;">Features</h2>
<ul style="line-height:1.7; font-size:16px;">
    <li>Detect emotions from images using a trained CNN</li>
    <li>Separate modules for dataset, face detection and augmentation</li>
    <li>Easy to extend or retrain with a different dataset</li>
    <li>Simple command-based execution</li>
</ul>

<h2 style="margin-top:30px;">Tech Used</h2>
<ul style="line-height:1.7; font-size:16px;">
    <li>Python</li>
    <li>TensorFlow / Keras</li>
    <li>OpenCV</li>
    <li>NumPy</li>
</ul>

<h2 style="margin-top:30px;">Project Structure</h2>
<pre style="background:#f6f6f6; padding:15px; border-radius:8px; font-size:15px;">
Emotion-Detection/
│
├── emotion_classifier/
│   └── emotionclassifier.py
│
├── emotion_det_files/
│
├── src_files/
│   ├── cnn.py
│   ├── dataset.py
│   ├── face_rec.py
│   ├── data_augmentation.py
│   └── main.py
</pre>

