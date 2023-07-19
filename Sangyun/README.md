# DACON Building Segmentation

## Repository Structure
---
``` bash
.<project_dir>/
├───.vscode
├───data/                       # Dacon Data Folder
│   ├───test_img/
│   ├───train_img/
│   ├───sample_submission.csv
│   ├───submit.csv
│   ├───test.csv
│   ├───train_edited.csv
│   └───train.csv
├───models/                     # Trained models saved to this folder
├───src/                         
│   ├───common/                 # Utilities
│   ├───dataset/                # Dataset Class
│   ├───model/                  # Backbone models
│   ├───__init__.py
│   ├───config.yaml             # Hyperparameter configurations
│   ├───main.py                 # Train / Inference baseline code
│   ├───pretrained.py           # Code for pretrained (deprecated)
└───README.md
```
