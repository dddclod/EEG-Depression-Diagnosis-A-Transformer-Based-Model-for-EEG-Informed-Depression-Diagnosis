# EEG-Depression-Diagnosis: A Transformer-Based Model for EEG-Informed Depression Diagnosis  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)  


## Project Overview  
This project addresses the core pain points of **strong subjectivity and low accuracy** in traditional scale-based assessment for clinical depression diagnosis. Taking electroencephalogram (EEG) signals as the core diagnostic carrier, it innovatively designs a dual-stream hybrid deep learning architecture of "Transformer Spectral Stream + 3D DenseNet Spatiotemporal Stream" to achieve objective and accurate automatic classification of depression .  

The project overcomes the issues of "spatiotemporal-frequency feature fragmentation" and "insufficient model generalization" in traditional EEG analysis. On the 128-channel EEG data from Lanzhou University’s MODMA dataset, it achieves a **classification accuracy of 87.5%**, which is 0.63 percentage points higher than mainstream models such as SST-EmotionNet. Meanwhile, it solves the "black box" problem of deep learning through attention weight visualization, providing technical support for early screening and clinical auxiliary diagnosis of depression .  


## Core Features  
1. **EEG Data Preprocessing**: Provides standardized batch processing scripts based on MATLAB EEGLab, supporting 0.1-40Hz band-pass filtering, ICA artifact removal, and bad channel/segment cleaning to significantly reduce the cost of manual data cleaning .  
2. **Dual-Stream Model Training**: Implements end-to-end training of the Transformer Spectral Stream (capturing global frequency-domain correlations) and 3D DenseNet Spatiotemporal Stream (extracting local spatiotemporal features), with automatic class weight balancing and early stopping mechanisms .  
3. **Model Performance Evaluation**: Incorporates multi-dimensional evaluation metrics (accuracy, precision, recall, F1-score) and automatically generates confusion matrices and training curves .  
4. **Attention Visualization**: Offers tools for generating brain region heatmaps and frequency band weight curves, intuitively demonstrating the model’s decision-making basis and establishing an interpretive link between "model features, neuropathological mechanisms, and clinical symptoms" .  
5. **Comparative Experiment Support**: Includes baseline implementations of mainstream models (CNN, GCN, LSTM, SST-EmotionNet) for quick comparative validation .  


## Environment Setup  
### Dependencies  
| Tool/Library          | Version Requirement | Description                          |  
|-----------------------|---------------------|--------------------------------------|  
| Python                | 3.8+                | Basic development environment        |  
| TensorFlow/PyTorch    | 2.8+/1.11+          | Deep learning framework (either one) |  
| MATLAB                | R2021b+             | For EEG data preprocessing (depends on EEGLab) |  
| EEGLab                | 2023.1+             | EEG signal processing toolbox        |  
| NumPy/Pandas          | 1.21+/1.4+          | Data storage and numerical computing |  
| Matplotlib/Seaborn    | 3.4+/0.11+          | Result visualization                 |  
| Scikit-learn          | 1.0+                | Data splitting and metric calculation |  

### Installation Steps  
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/EEG-Depression-Diagnosis.git
   cd EEG-Depression-Diagnosis
   ```  
2. Install Python dependencies  
   ```bash
   pip install -r requirements.txt
   ```  
3. Install EEGLab (MATLAB): Follow the tutorial on the [EEGLab official website](https://sccn.ucsd.edu/eeglab/index.php) and place the `preprocess/matlab/eeg_preprocess.m` script in the EEGLab working directory .  


## Dataset Description  
The core experimental data of this project is based on **Lanzhou University’s MODMA (Multi-modal Open Dataset for Mental Disorder Analysis) dataset**—the first domestic multi-modal clinical dataset for mental disorder analysis, meeting the three requirements of "clinically confirmed labels, standardized collection, and multi-task states" .  

### Key Dataset Information  
| Indicator              | Details                                                                 |  
|------------------------|-------------------------------------------------------------------------|  
| Data Source            | Second Hospital of Lanzhou University (clinical depression patients) + social recruitment (healthy controls), compliant with the Declaration of Helsinki  |  
| Subject Scale          | 53 valid subjects (24 depressed patients, 29 healthy controls, aged 16-56 years)  |  
| Diagnostic Criteria    | Patients confirmed via MINI interview (meeting DSM-IV criteria for major depression), PHQ-9 score ≥5, no psychiatric medication in the past 2 weeks  |  
| Data Specification     | 128-channel EEG, 500Hz sampling rate, including resting-state (5 minutes with eyes closed) and dot-probe task-state data  |  

### Data Acquisition and Preprocessing  
1. Apply for the MODMA dataset: Visit the [Lanzhou University MODMA Dataset official website](https://modma.lzu.edu.cn/) and submit an application to obtain raw data .  
2. Data preprocessing: Run the MATLAB script `preprocess/matlab/eeg_preprocess.m` to automatically perform:  
   - 0.1-40Hz band-pass filtering  
   - ICA artifact removal (ocular and myoelectric interference)  
   - Whole-brain average re-referencing  
   - Bad channel/segment removal and baseline correction  
   - Downsampling (optional, 500Hz retained by default)   
3. Data format conversion: Run `preprocess/python/convert_mat2npy.py` to convert preprocessed MAT files to NPY format (readable by the model) and store them in `data/processed/` .  


## Model Architecture  
The project adopts a **dual-stream heterogeneous deep learning architecture**, designed with dedicated modules for the different characteristics of the "temporal-frequency-spatial" three-dimensional features of EEG signals:  

### 1. Transformer Spectral Stream  
- **Input**: 20×20×5×1 (spatial grid × 5 frequency bands × channel)  
- **Core Function**: Captures global correlations of Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), and Gamma (30-50Hz) bands .  
- **Key Components**:  
  - Spatial Embedding Layer: Flattens the 20×20 spatial dimension into 400 sequences while retaining frequency band information .  
  - Learnable Positional Encoding: Preserves the spatial topological relationships of brain regions .  
  - Multi-Head Self-Attention (4 heads, 64 embedding dimensions): Models global dependencies across frequency bands and brain regions .  
  - Global Average Pooling: Outputs a 64-dimensional frequency-domain feature vector .  

### 2. 3D DenseNet Spatiotemporal Stream  
- **Input**: 20×20×512×1 (spatial grid × 512 time points × channel)  
- **Core Function**: Extracts the spatial topological structure and temporal dynamic patterns of EEG signals .  
- **Key Components**:  
  - Decomposed 3D Convolution: (3×3×1) spatial convolution + (1×1×3) temporal convolution, reducing parameters by 1/3 .  
  - Dense Blocks (2 units): Reuses inter-layer features to enhance local detail capture .  
  - Spatial-Temporal Dual Attention: Dynamically focuses on key brain regions (e.g., prefrontal lobe) and emotion-related time segments .  
  - Global Average Pooling: Outputs a high-dimensional spatiotemporal feature vector .  

### 3. Feature Fusion and Classification  
- Fusion Strategy: Dimension-wise concatenation of the 64-dimensional spectral stream features and spatiotemporal stream features .  
- Classification Layer: Fully connected layers (128→64) + Dropout (0.5) + Softmax (2-class output: healthy/depressed) .  


## Quick Start  
### 1. Data Preparation  
Store preprocessed NPY-format data in the following structure:  
```
data/
├── processed/
│   ├── train/
│   │   ├── healthy/      # Healthy control samples (NPY files)
│   │   └── depressed/    # Depressed patient samples (NPY files)
│   ├── val/              # Validation set (same structure as above)
│   └── test/             # Test set (same structure as above)
```  

### 2. Model Training  
Run the training script, supporting custom hyperparameters (e.g., learning rate, number of epochs):  
```bash
# Basic training (default parameters)
python train.py --config configs/default.yaml

# Custom training (example)
python train.py --config configs/custom.yaml \
                --lr 0.001 \
                --epochs 100 \
                --batch_size 16
```  
- Training logs and model weights are automatically saved to the `runs/` directory .  
- Real-time visualization of training curves (accuracy/loss) via TensorBoard:  
  ```bash
  tensorboard --logdir runs/
  ```  

### 3. Model Evaluation  
Evaluate model performance on the test set and generate confusion matrices and metric reports:  
```bash
python evaluate.py --model_path runs/20241007_best_model.h5 \
                   --test_data_path data/processed/test/
```  
- Evaluation results are saved to the `results/` directory, including:  
  - Numerical report of metrics (accuracy, precision, recall, F1-score) (`metrics.txt`) .  
  - Confusion matrix visualization (`confusion_matrix.png`) .  
  - Attention weight heatmaps (`attention_heatmap/`) .  

### 4. Baseline Model Comparison  
Run the comparative experiment script to verify performance differences between the proposed model and mainstream models:  
```bash
python baseline_comparison.py --models cnn gcn lstm sst-emotionnet
```  
- Comparison results are saved to `results/baseline_comparison.csv`, with optional bar chart generation (`baseline_plot.png`) .  


## Core Innovations  
1. **Transformer for EEG Spectral Analysis**: For the first time, Transformer is applied to EEG frequency-domain feature extraction, breaking the local receptive field limitation of traditional convolution and enabling one-step global correlation modeling of whole-brain frequency bands .  
2. **Dual-Stream Heterogeneous Architecture**: The spectral stream (Transformer) and spatiotemporal stream (3D DenseNet) complement each other, capturing global frequency-domain correlations and local spatiotemporal patterns respectively to achieve deep fusion of "temporal-frequency-spatial" features .  
3. **Decomposed 3D Convolution and Dual Attention**: Splits 3D convolution into independent spatial and temporal operations, reducing parameters by 1/3; combines spatial-temporal attention to dynamically focus on key brain regions and time segments .  
4. **Small-Sample Adaptation Strategy**: Addresses small-sample imbalance through class weight balancing (healthy: 0.925, depressed: 1.088), data augmentation (flip/rotation/noise, 4x sample size expansion), and early stopping .  


## Experimental Results  
### 1. Model Performance (MODMA Test Set)  
| Metric         | Proposed Model | SST-EmotionNet | CNN     | GCN     | LSTM    | EEGNet  |  
|----------------|----------------|----------------|---------|---------|---------|---------|  
| Accuracy       | 0.8750         | 0.8125         | 0.4463  | 0.5707  | 0.5481  | 0.7656  |  
| Precision      | 0.9028         | 0.8146         | 0.3711  | 0.2854  | 0.0000  | 0.7659  |  
| Recall         | 0.8750         | 0.8125         | 0.4170  | 0.5000  | 0.0000  | 0.7586  |  
| F1-Score       | 0.8750         | 0.8102         | 0.3927  | 0.3634  | 0.0000  | 0.7600  |  

### 2. Key Advantages  
- **No Missed Diagnoses**: The confusion matrix shows 100% recognition rate of depressed samples, meeting the clinical need for "avoiding missed diagnoses" in early screening .  
- **Interpretability**: Attention heatmaps intuitively display key brain regions (e.g., prefrontal lobe, parietal lobe) focused on by the model, and frequency band weight curves can correlate Theta/Alpha wave abnormalities (physiological markers of depression) .  
- **Generalization**: Maintains stable performance with small samples (53 cases), controlling overfitting within 3% (lower than the 5%-8% of similar models) .  


## Project Structure  
```
EEG-Depression-Diagnosis/
├── configs/                # Configuration files (model parameters, data paths, etc.)
│   ├── default.yaml
│   └── custom.yaml
├── data/                   # Data directory (raw/preprocessed data)
│   ├── raw/
│   └── processed/
├── models/                 # Model definitions
│   ├── transformer_spectral.py  # Transformer Spectral Stream
│   ├── densenet_temporal.py     # 3D DenseNet Spatiotemporal Stream
│   ├── fusion_model.py          # Dual-stream fusion model
│   └── baselines/               # Baseline models (CNN/GCN/LSTM, etc.)
├── preprocess/             # Data preprocessing scripts
│   ├── matlab/
│   │   └── eeg_preprocess.m     # MATLAB EEG preprocessing
│   └── python/
│       └── convert_mat2npy.py   # Format conversion
├── runs/                   # Training logs and model weights
├── results/                # Evaluation results and visualizations
├── utils/                  # Utility functions
│   ├── attention_visualization.py  # Attention visualization
│   ├── metrics.py                  # Metric calculation
│   └── data_augmentation.py        # Data augmentation
├── train.py                # Main model training script
├── evaluate.py             # Main model evaluation script
├── baseline_comparison.py  # Baseline model comparison script
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation
```  


## Limitations and Future Directions  
### Limitations  
1. **Feature Dimension Fragmentation**: Although the model fuses temporal-frequency-spatial features, it does not establish an intrinsic correlation model for 3D information, making it difficult to fully restore the global abnormal features of brain neural activity in depressed patients .  
2. **Insufficient Data Scale**: The MODMA dataset has a small sample size (53 cases) and limited population representativeness (mainly middle-aged and young adults), lacking multi-center and multi-modal synchronous data .  
3. **Needs Improved Generalization**: The model is trained on a single dataset and may perform poorly when transferred to scenarios with comorbid physical diseases or elderly patients .  

### Future Directions  
1. **Enhance Interpretability**: Develop interactive visualization tools to establish a quantitative link between model features and PHQ-9 scale/MINI interview results .  
2. **Expand Data Support**: Collaborate with multiple centers to build a dataset of over 800 cases, including adolescent/elderly patients and comorbid samples, and supplement 3-electrode EEG data from wearable devices .  
3. **Optimize Generalization**: Introduce individual calibration mechanisms and domain-adaptive learning to improve model stability across datasets and populations .  


## Patent and Citation  
The core technology of this project has been applied for an invention patent: *"Research on EEG Depression Diagnosis Algorithm Based on Multi-Dimensional Attention Mechanism"* .  

If you use the code or model of this project in your research, please cite the following related work:  
1. Sun, Y., et al. (2019). MODMA: A Multi-modal Open Dataset for Mental Disorder Analysis. *Computers in Biology and Medicine*.  
2. Project Team. (2024). EEG-Depression-Diagnosis: A Transformer-DenseNet Hybrid Model for EEG-Based Depression Diagnosis. *GitHub Repository*.  


## Acknowledgments  
- Thanks to Lanzhou University for providing the MODMA dataset, which serves as a key support for model training and validation .  
- Thanks to the developers of tool platforms such as EEGLab and TensorFlow for providing technical foundations for EEG processing and deep learning .  
- Thanks to all project team members for their collaboration and contributions in model design, experimental validation, and documentation .  


## License  
This project is open-source under the MIT License. See the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute the project code, but you must retain the original copyright notice and license information.
