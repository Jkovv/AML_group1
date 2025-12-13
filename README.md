### The outline: 
we created a preprocessing pipeline, utilized a base model (MobileNetV2 from Hugging Face), created a hybrid model (Convolution + Attention) and a few simpler models for context (CNN, VIT, ResNet).

Files:

## 1. `data_prep.ipynb`
The Data Engineering pipeline notebook. Converts all images to RGB (fixing grayscale errors) and standardizes file paths, fixes class labels (1-etc to 0-...), and saves the dataset Apache Arrow format (`processed_bird_data_FULL`) for instant loading. The notebook then also splits this set into Training and Validation sets (85/15 split, but easy to change) in the folder `processed_bird_data`. Finally, test set is also saved in that format (`processed_bird_test_data`).

The processed files are too large for GitHub, so they are stored on OneDrive - [OneDrive](https://amsuni-my.sharepoint.com/:f:/g/personal/rezi_getsadze_student_uva_nl/IgAYqAmSNJUiQaxMoPGTHkanARL3nATHglMr0nz9Y8AF-G4?e=Yx6cxA).

To use them, place the folder in the project root, and use this code to load the data and create the Validation Set:

```python
from datasets import load_from_disk

# Load the Training & Validation sets (DatasetDict)
dataset = load_from_disk("processed_bird_data")

# Usage:
# dataset["train"]
# dataset["validation"]

# For test set:
test_dataset = load_from_disk("processed_bird_test_data")
```

## 2. `baseline_model.ipynb`

Baseline/benchmark model, using Transfer Learning.

Loads Pre-trained Model (MobileNetV2) using pretrained weights; retrains/finetunes for **5 epochs**.

Achieves 55-60% accuracy (saved in `baseline_model` folder).

## 3. `model_coatnet.ipynb`

New hybrid model (Convolution + Attention) with 25% accuracy.

Output model file yet too large, so uploaded to OneDrive (`new_model` folder) — [OneDrive](https://amsuni-my.sharepoint.com/:f:/r/personal/rezi_getsadze_student_uva_nl/Documents/AML?csf=1&web=1&e=B0Slia).

Uses optuna to optimize for best hyperparameters. Model is created with random seed 42, and then seeds 1 and 100 are checked (and if they are better, they are saved).

## 4. `model_CNN.ipynb`
An attempt at CNN - for comparability with the main model.

## 5. `model_ResNet.ipynb`
An attempt at ResNet - for comparability with the main model.

## 6. `model_convext.ipynb`
ConvNeXt model attempt (title up for discussion) - for comparability with the main model (9% best accuracy).

## 7. `model_MAE_ViT.ipynb`
ViT, but nuclear option - uses masked autoencoders to pre-train model on the images themselves without labels.

## 8. `old_model_coatnet.ipynb`

Coatnet version without optimization, directly with initially assumed parameters. Might be useful for comparison/hyperparameter study.
