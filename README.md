# Distracted Driver Detection

UE24CS352A Machine Learning mini-project.

## Problem statement
Classify an image taken inside a car into one of 10 classes: safe driving
(c0) and nine distraction activities (texting with the right or left hand,
talking on the phone with the right or left hand, operating the radio,
drinking, reaching behind, hair and makeup, talking to a passenger).

## Dataset
Kaggle: State Farm Distracted Driver Detection, 22,424 labelled images of
26 drivers across classes c0 to c9. The dataset is not stored in this
repo. Download it and place it at:

    data/train/c0 ... data/train/c9
    data/driver_imgs_list.csv

Do not include the unlabelled test folder.

## Approach
- Each image is converted to RGB, resized to 64x64x3 and flattened to a
  12,288-dimensional vector. Pixels are scaled to [0,1] and the training
  mean is subtracted.
- The data is split 80/20 **by driver** (17,446 train / 4,978 validation
  images), so no driver appears in both sets.
- Five classifiers: Linear SVM, Softmax regression, Gaussian Naive Bayes,
  Decision Tree and a two-layer neural network (100 ReLU hidden units,
  softmax output).

## Setup
    pip install -r requirements.txt

## How to run (in this order)
The dataset, processed arrays and trained models are git-ignored, so on a
fresh clone you must run preprocessing and training before the demo.

    python src/preprocess.py
    python src/train_naive_bayes.py
    python src/train_decision_tree.py
    python src/train_svm.py
    python src/train_softmax.py
    python src/train_nn.py
    python src/train_nn_tuned.py
    python src/evaluate.py
    python src/random_split_check.py
    python src/make_val_samples.py
    python demo/app.py

`evaluate.py` writes the results table, per-class accuracy and confusion
matrix to `results/`. `make_val_samples.py` copies held-out-driver images
to `demo/val_samples/` for testing the demo. `demo/app.py` starts a Gradio
web app at http://127.0.0.1:7860 that loads the best model (by validation
accuracy) and classifies an uploaded image.

## Results (driver-wise validation split)
| Model | Train Acc. (%) | Val Acc. (%) |
|---|---|---|
| Linear SVM | 95.25 | 33.69 |
| Softmax | 95.93 | 37.46 |
| Naive Bayes | 56.06 | 27.56 |
| Decision Tree | 98.88 | 21.90 |
| Two-layer NN | 100.00 | 38.87 |
| Two-layer NN (tuned, alpha=1.0) | 98.39 | 40.60 |

Same two-layer NN with a **random** (non-driver-wise) split: train 100.00,
val 99.71.

### Findings
- All models fit the training drivers well but generalise poorly to unseen
  drivers. Flattened pixels without convolutions do not transfer across
  people.
- The gap between the random split (99.71%) and the driver-wise split
  (40.60%) shows that random splits leak driver identity and
  near-identical consecutive frames, so they overestimate real
  performance.
- Best classes (tuned NN): hair and makeup 82.25%, talking to passenger
  62.84%, radio 57.98%. Worst: safe driving 11.52%.
- Regularization was tuned on the validation set, so 40.60% is slightly
  optimistic. There is no separate test set of drivers.

## Future work
Convolutional or pretrained models (ResNet, VGG), data augmentation, and a
separate driver-wise test set.

## Team
Both members developed the code together and committed it from one
laptop after the full pipeline and demo were working.
- **Spoorthi Poornachandra (PES1UG24AM279):** preprocessing, linear SVM, softmax, evaluation, demo code, write-up sections 1-4, slides 1-5
- **Yashwanth Gowda T G (PES1UG25AM818):** Naive Bayes, decision tree, neural networks, random-split check, validation samples, write-up sections 5-6, slides 6-10