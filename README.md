# kaggle-petfinder-adoption-prediction

Project of Team A at the University of Szeged Machine learning course 2026

In this project we aim to solve the kaggle petfinder problem using machine learning solutions.

## Dataset

### Purpose and content
This project uses the **PetFinder Adoption Prediction** competition dataset to predict `AdoptionSpeed` (how fast a listed pet is adopted, classes `0-4`).

The dataset combines:
- tabular + text listing fields (for example age, breed, health, fee, and profile description),
- optional images,
- optional Google Vision image metadata,
- optional Google Natural Language sentiment/entity outputs.

### Unit of observation (records)
Each row in the main CSV files represents one **PetFinder listing/profile** (`PetID`).
Some listings represent a group of pets (`Quantity > 1`); in those cases, the label reflects adoption speed for the whole listed group.

### Source and provenance
- Kaggle competition page (primary source): https://www.kaggle.com/c/petfinder-adoption-prediction
- Data access path used in this repo: see `notebooks/petfinder_setup_guide.md` (Kaggle API setup + rules acceptance)
- Original data provider (via competition): **PetFinder.my** (Malaysia pet adoption platform)

### Relevant input files and roles
- `train/train.csv`: training tabular/text data with target `AdoptionSpeed`
- `test/test.csv`: test tabular/text data without `AdoptionSpeed` (prediction target)
- `test/sample_submission.csv`: required Kaggle submission format
- `breed_labels.csv`: lookup for breed IDs (+ pet type)
- `color_labels.csv`: lookup for color IDs
- `state_labels.csv`: lookup for Malaysia state IDs
- `train_images/`, `test_images/`: pet photos (`PetID-ImageNumber.jpg`)
- `train_metadata/`, `test_metadata/`: Google Vision metadata (`PetID-ImageNumber.json`)
- `train_sentiment/`, `test_sentiment/`: Google Natural Language outputs (`PetID.json`)

### Dataset size and split used in this project
Using the competition Stage 1 files loaded in `notebooks/petfinder_data_quality_and_features.ipynb`:
- training set: **14,993 rows × 24 columns** (includes `AdoptionSpeed`)
- test set: **3,972 rows × 23 columns** (no `AdoptionSpeed`)

Project workflows follow the official Kaggle split (`train/train.csv` for modeling and `test/test.csv` for final predictions/submission).
