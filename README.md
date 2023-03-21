# Cross-modal Multiscale Multi-instance Learning for Long-term ECG Classification
 This is the code for the paper "Cross-modal Multiscale Multi-instance Learning for Long-term ECG Classification"
 
# Dependency
- python>=3.7
- pytorch>=1.11.0
- torchvision>=0.12.0
- numpy>=1.21.5
- tqdm>=4.64.0
- scipy>=1.7.3
- wfdb>=3.4.1
- scikit-learn>=1.0.2
 
# Usage
## Configuration
There is a configuration file "config.py", where one can edit both the training and test options.
 
## Stage 1: Data Process
First, data processing is required, simply run
python data_process.py

## Stage 2: Training
 After setting the configuration, to start training, simply run
 python main.py
 
# Dataset
St. Petersburg INCART Arrhythmia dataset can be downloaded from
https://www.physionet.org/content/incartdb/1.0.0/

MIT-BIH Arrhythmia dataset can be downloaded from
https://www.physionet.org/content/mitdb/1.0.0/

# Citation
If you find this idea useful in your research, please consider citing: "Cross-modal Multiscale Multi-instance Learning for Long-term ECG Classification"
