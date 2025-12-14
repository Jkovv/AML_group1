## Project Outline
A complete machine learning pipeline for bird species classification.
* **Data Engineering:** Preprocessing pipeline (RGB conversion, label fixing, Arrow format).
* **Baseline:** Pre-trained **MobileNetV2** (Hugging Face) for benchmarking.
* **Main Model:** Custom **CoAtNet** (Hybrid Convolution + Attention) trained from scratch.
* **Context Models:** Simple CNNs, ResNet, ConvNeXt, and MAE (ViT) for comparative analysis.

## Files

### 1. Data Engineering
* **`data_prep.ipynb`** - ETL pipeline. Standardizes paths, fixes grayscale images, and corrects labels. Applies data split adn saves in Apache Arrow format for instant loading. Obviously those files are too big to upload here so they're at [OneDrive](https://amsuni-my.sharepoint.com/:f:/g/personal/rezi_getsadze_student_uva_nl/IgAYqAmSNJUiQaxMoPGTHkanARL3nATHglMr0nz9Y8AF-G4?e=Yx6cxA).
TO use, this can be used in the model codes:
        ```python
        from datasets import load_from_disk
        dataset = load_from_disk("processed_bird_data") # Train/Val
        test_ds = load_from_disk("processed_bird_test_data") # Test
        ```

### 2. Baseline Benchmark
* **`baseline_model.ipynb`** - Pre-trained MobileNetV2, with saved `baseline_best_model.pth`.

### 3. Main Architecture
* **`model_coatnet.ipynb`** - Custom **CoAtNet** (Hybrid) trained from scratch. Uses Optuna hyperparameter tuning, builda model, and does seed stability checks. Final model's on OneDrive as well, in `final_new_model` folder - [OneDrive](https://amsuni-my.sharepoint.com/:f:/r/personal/rezi_getsadze_student_uva_nl/Documents/AML?csf=1&web=1&e=B0Slia). Checkpoint model saves from seed stability runs are also located in the folder `seedcheckpoints`. The optimization checkpoints are available in `dec13modelcheckpoints`. The file also handles the test batch and outputs Kaggle submission csv `coatnet_submission.csv`.

### 4. Comparative Models [validation accuracy]
* **`model_CNN.ipynb`**: Standard CNN for baseline comparison (~8% accuracy).
* **`model_ResNet.ipynb`**: ResNet implementation (~15% accuracy).
* **`model_convnext.ipynb`**: ConvNeXt implementation (~9% accuracy).
* **`model_MAE_step1_pretrain.ipynb`** & **`_step2_finetune.ipynb`**: Self-supervised pre-training using **Masked Autoencoders (ViT)**.

### 5. Legacy
* **`old_model_coatnet.ipynb`**: Pre-optimization version of the main model (will be used for comparison). The saved models/checkpoints available in the `old_hyperparam_model_checlpoints` folder - [OneDrive](https://amsuni-my.sharepoint.com/:f:/r/personal/rezi_getsadze_student_uva_nl/Documents/AML?csf=1&web=1&e=B0Slia). The CSV generation was made in a separate file `old_model_csv_generation.ipynb`.
* **`model_coatnet_augmentation.ipynb`** - test/experiment file in an attempt to reduce coanet overfitting. augments the images to stop model from memorizing. Output file `coatnet_augmented_submission.csv`. Final model can be found in the `coatnet_aug_experiment\best_aug_model` folder - [OneDrive](https://amsuni-my.sharepoint.com/:f:/r/personal/rezi_getsadze_student_uva_nl/Documents/AML?csf=1&web=1&e=B0Slia).
* 

### 6. Error Analysis
`erroranalysis.ipynb` - Diagnostic notebook designed to interpret model performance beyond simple accuracy metrics.
* **Top Losses Visualization:** Displays test images with the highest loss, highlighting instances where the model was confidently incorrect (e.g., confusing visually similar species).
* **Confusion Matrix:** Heatmaps to identify specific clusters of bird species that the model frequently misclassifies.


