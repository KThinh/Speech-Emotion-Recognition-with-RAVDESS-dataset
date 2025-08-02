# Speech-Emotion-Recognition-with-RAVDESS-dataset
3D CNN for speech emotion recognition on the RAVDESS dataset using MFCC &amp; mel spectrogram features.

This project implements a robust 3D Convolutional Neural Network (3D CNN) for speech emotion recognition using the RAVDESS dataset. The pipeline includes audio preprocessing, data augmentation (noise injection, pitch shifting), class imbalance handling, MFCC + mel spectrogram feature extraction, and L2 normalization to reduce speaker bias.

The model achieves over **95% test accuracy** and a macro F1-score of **0.95+** across 8 emotion classes, demonstrating strong generalization.

Key highlights:
- Conv3D architecture with GroupNorm, Dropout3D, and global pooling
- Data augmentation & oversampling for handling label imbalance
- L2 normalization to reduce overfitting on speaker identity
- Early stopping and detailed training visualization
- Classification report with precision, recall, and F1-score per class

## Limitations

While the model achieves high accuracy (95%+) on the RAVDESS dataset, it's important to note that RAVDESS is a small, clean, and highly acted dataset with limited diversity in speakers, recording conditions, and naturalness of emotions.

Therefore, this model may not generalize well to real-world audio with spontaneous emotions, background noise, varied accents, or different recording environments. Further training on larger, more diverse datasets (e.g., CREMA-D, EmoDB, or real-world recordings) and domain adaptation would be required for practical deployment.
