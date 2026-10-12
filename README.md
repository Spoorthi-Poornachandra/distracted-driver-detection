# distracted-driver-detection
Distracted driver detection from dashboard images (State Farm dataset) using Linear SVM, Softmax, Naive Bayes, Decision Tree and a two-layer neural network. UE24CS352A ML mini-project.
# Distracted Driver Detection

UE24CS352A Machine Learning mini-project.

## Problem statement
Classify driver images into 10 classes (safe driving + 9 distraction
activities) from the State Farm Distracted Driver Detection dataset.

## Dataset
Kaggle: State Farm Distracted Driver Detection (22,424 labelled images,
classes c0 to c9). Download the `train` folder and place it at:

    data/train/c0 ... data/train/c9
    data/driver_imgs_list.csv

## Setup
    pip install -r requirements.txt

## How to run
    python src/preprocess.py
    python src/train_svm.py
    python src/train_softmax.py
    python src/train_naive_bayes.py
    python src/train_decision_tree.py
    python src/train_nn.py
    python src/evaluate.py
    python demo/app.py

## Results
| Model | Train Acc. | Val Acc. |
|---|---|---|
| Linear SVM | | |
| Softmax | | |
| Naive Bayes | | |
| Decision Tree | | |
| Two-layer NN | | |

## Team
- Name 1 (SRN): what they did
- Name 2 (SRN): what they did