# Multimodal EEG-IMU Fusion for Motor Assessment

Code for the paper: **"Task-Dependent Performance Patterns Enable Robust Multimodal Motor Assessment: Integrating EEG and IMU Signals for Reliable Movement Classification"**

## Overview

This repository implements a multimodal fusion system combining EEG (brain signals) and IMU (motion sensors) for motor activity classification, achieving **98.68% accuracy**.

## Key Results

- **EEG only**: 92.82% accuracy
- **IMU only**: 94.41% accuracy  
- **Fusion**: 98.68% accuracy
- **Worst-task accuracy improved**: 87% → 97%

## Installation
```bash
git clone https://github.com/iupui-soic/har-eeg.git
cd har-eeg
pip install -r requirements.txt
```

## Model Architecture

### EEG Branch: EEGNet + Transformer
- 2 transformer layers, 8 attention heads
- Embedding dimension: 128
- Dropout: 0.5 (EEGNet), 0.3 (Transformer)
- Training: Adam (lr=0.001), batch size=32

### IMU Branch: XGBoost
- 152 features → Top 60 by importance
- Max depth: 6, learning rate: 0.1
- 100 estimators

### Fusion
- Late fusion via logistic regression
- Trained on validation set predictions

## Citation
```bibtex
@article{yin2026multimodal,
  title={Task-Dependent Performance Patterns Enable Robust Multimodal Motor Assessment},
  author={Yin, Zhenan and Pulavarthy, Lalitha Pranathi and Purkayastha, Saptarshi},
  year={2026}
}
```

## Contact

- Zhenan Yin - yin10@iu.edu
- Indiana University Indianapolis

## License

MIT License
