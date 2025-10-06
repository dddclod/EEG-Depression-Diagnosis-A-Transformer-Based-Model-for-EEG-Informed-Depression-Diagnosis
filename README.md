# EEG-Depression-Diagnosis: Transformer - Based EEG Depression Diagnosis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)

## Project Overview
This project targets the clinical depression diagnosis pain points of strong subjectivity and low accuracy in traditional scale assessment. Taking electroencephalogram (EEG) signals as the diagnostic medium, it innovatively designs a dual - stream hybrid deep learning architecture combining a Transformer spectral stream and a 3D - based spatiotemporal stream to achieve objective and accurate automatic classification of depression.

The project breaks through the limitations of traditional EEG analysis in terms of fragmented spatiotemporal - frequency features and insufficient model generalization. It achieves good classification results on EEG data and solves the "black box" problem of deep learning through attention weight visualization, providing technical support for early screening and clinical auxiliary diagnosis of depression.

## Core Features
1. **EEG Data Preprocessing**: Capable of performing operations such as filtering, artifact removal, and re - referencing on EEG data to prepare high - quality data for model training.
2. **Dual - Stream Model Training**: Enables end - to - end training of the Transformer spectral stream (for capturing global frequency - domain correlations) and the spatiotemporal stream (for extracting local spatiotemporal features), with support for strategies like automatic class weight balancing.
3. **Comprehensive Model Evaluation**: Generates multiple evaluation results including classification reports, confusion matrices, and training history metrics to fully assess model performance.
4. **Attention Visualization**: Provides visualization tools for attention weights, intuitively showing the model's focus on key brain regions and signal segments, and establishing a link between model features and clinical symptoms.
5. **Model Saving and Loading**: Supports saving trained models for subsequent inference and comparison, and also allows loading pre - trained models for direct use.

## Environment Setup
### Dependencies
| Tool/Library          | Version Requirement | Description                          |
|-----------------------|---------------------|--------------------------------------|
| Python                | 3.8+                | Basic development environment        |
| TensorFlow            | 2.8+                | Deep learning framework              |
| NumPy                 | Latest              | Numerical computing                  |
| Matplotlib            | Latest              | Result visualization                 |
| Scikit - learn        | Latest              | Metric calculation and data handling |

### Installation Steps
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/EEG-Depression-Diagnosis.git
   cd EEG-Depression-Diagnosis
   ```
2. Install Python dependencies (using pip, example command, actual dependencies may vary)
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Description
The project uses EEG datasets suitable for depression diagnosis research. The data should undergo preprocessing steps such as filtering to remove noise and artifact removal to eliminate interference from eye movements, muscle activity, etc. The preprocessed data is then prepared into a format suitable for model input.

## Model Architecture
### Transformer Spectral Stream
- **Function**: Focuses on capturing the global correlation information in the frequency domain of EEG signals.
- **Key Components**: Utilizes Transformer's self - attention mechanism to model the relationships between different frequency components across the entire brain.

### Spatiotemporal Stream
- **Function**: Aims to extract local spatiotemporal feature patterns from EEG signals.
- **Key Components**: Uses 3D - based operations to capture the spatial distribution and temporal dynamics of EEG signals.

### Fusion and Classification
The features from the two streams are fused and then passed through a classification layer to obtain the final depression diagnosis result.

## Quick Start
### Data Preparation
Prepare preprocessed EEG data and organize it into a structure that the model can read. Ensure the data is split into training, validation, and test sets as needed.

### Model Training
Run the training script to start training the dual - stream model:
```bash
python run.py
```
During training, configuration files like `config.ini` and `config_transformer.ini` are used to set model and training parameters. The `create_labels.py` script can be used to generate label files for the data.

### Model Evaluation
After training, the evaluation results are saved in the `results` directory. You can find:
- `all_result.txt`: Comprehensive result summary.
- `classification_report.txt`: Detailed classification report including precision, recall, and F1 - score.
- `confusion_matrix.png`: Visualization of the confusion matrix.
- `metrics_table.png`: Visual table of evaluation metrics.
- `training_history.png` and `training_history.txt`: Training process metrics such as loss and accuracy over epochs.

### Model Inference
Use the saved model (in the `model` directory under `results`) for inference on new data. You can also use scripts like `测试文件.py` (ensure it's for legitimate testing purposes) for specific testing tasks.

## Project Structure
```
EEG-Depression-Diagnosis/
├── results/                # Directory for storing result files
│   ├── all_result.txt      # Comprehensive result summary
│   ├── classification_report.txt # Detailed classification report
│   ├── confusion_matrix.png # Confusion matrix visualization
│   ├── metrics_table.png   # Visual table of evaluation metrics
│   ├── model/              # Saved model files
│   ├── training_history.png # Training history visualization (image)
│   ├── training_history.txt # Training history (text)
│   └── 测试文件.py          # Test script (for legitimate testing)
├── README.md               # Project description document
├── config.ini              # General configuration file
├── config_transformer.ini  # Transformer - related configuration file
├── create_labels.py        # Script for creating data labels
├── model.py                # General model definition script
├── model_transformer.py    # Transformer model definition script
├── run.py                  # Main script for running training and inference
└── requirements.txt        # Python dependency list
```

## Acknowledgments
- Thanks to the open - source communities of TensorFlow, NumPy, and other libraries for providing technical support.
- Acknowledge the sources of the EEG datasets used in the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
