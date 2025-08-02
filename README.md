# Speech-Emotion-Recognition-with-RAVDESS-dataset
This project implements a 3D Convolutional Neural Network (3D CNN) for speech emotion recognition using the RAVDESS dataset. The pipeline includes audio preprocessing, data augmentation (noise injection, pitch shifting), class imbalance handling, MFCC + mel spectrogram feature extraction, and L2 normalization to reduce speaker bias.

Key highlights:
- Conv3D architecture with GroupNorm, Dropout, and global pooling
- Data augmentation & oversampling for handling label imbalance
- L2 normalization to reduce overfitting on speaker identity
- Early stopping and detailed training visualization
- Classification report with precision, recall, and F1-score per class

## Model Performance
- **Training Accuracy:** ~99.1%
- **Validation Accuracy:** ~94.8%
- **Test Accuracy:** ~93.5%
- **Best Macro F1-score:** ~93.4%

## Limitations

While the model achieves high accuracy (93,5%) on the RAVDESS dataset, it's important to note that RAVDESS is a small, clean, and highly acted dataset with limited diversity in speakers, recording conditions, and naturalness of emotions.

Therefore, this model may not generalize well to real-world audio with spontaneous emotions, background noise, varied accents, or different recording environments. 
