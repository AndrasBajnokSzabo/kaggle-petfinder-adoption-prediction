# PetFinder features and data quality

All results come from `notebooks/petfinder_data_quality_and_features.ipynb`, run on `train/train.csv` (14,993 rows × 24 columns).

## Feature overview
pandas stores 19 of the 24 columns as `int64`, so the storage type does not show what a column means. The semantic types below were assigned manually from the Kaggle data description.

| Type | Variables | Meaning and encoding |
| :------ | :---------------- | :------------------------------------------------- |
| Target (ordinal) | `AdoptionSpeed` | 0: same day, 1: 1–7 days, 2: 8–30 days, 3: 31–90 days, 4: not adopted after 100 days |
| Binary | `Type` | 1: Dog, 2: Cat |
| Nominal | `Gender` | 1: Male, 2: Female, 3: Mixed (group profile) |
| | `Breed1`, `Breed2` | Breed ID from `breed_labels.csv` (307 breeds); `Breed2 = 0`: none |
| | `Color1`–`Color3` | Colour ID from `color_labels.csv` (7 colours); 0: none |
| | `State` | Malaysian state ID from `state_labels.csv` (15 states) |
| | `Vaccinated`, `Dewormed`, `Sterilized` | 1: Yes, 2: No, 3: Not Sure |
| Ordinal | `MaturitySize` | 1: Small → 4: Extra Large, 0: Not Specified |
| | `FurLength` | 1: Short → 3: Long, 0: Not Specified |
| | `Health` | 1: Healthy → 3: Serious Injury, 0: Not Specified |
| Numerical | `Age` | Age at listing, in months |
| | `Quantity` | Number of pets in the profile (group if > 1) |
| | `Fee` | Adoption fee, 0 = free |
| | `VideoAmt`, `PhotoAmt` | Number of uploaded videos / photos |
| Text | `Name`, `Description` | Pet name (empty if unnamed); profile text (mostly English, some Malay / Chinese) |
| Identifier | `PetID`, `RescuerID` | Hash IDs of the profile and of the rescuer; not used as features |

**Encoded values.**

* **`0` in `Breed2` / `Color2` / `Color3` means "none".** This is our inference: `0` is not an ID in the label files.
* **Some codes are never used.** `0 = Not Specified` never occurs, and one state (Perlis) has no listing.
* **Breed is dominated by non-pedigree categories.** *Mixed Breed* is 72.8% of the dogs, and *Domestic Short / Medium / Long Hair* is 75.6% of the cats. Overall 116 dog and 68 cat breed IDs occur, most of them rarely, so for M2 a pure-breed flag is more practical than one category per breed.
* **Colours are stored in ascending ID order.** Black (ID 1) never appears as `Color2` / `Color3`, so `Color1` is not necessarily the main colour.
* **Location and target are uneven.** Selangor and Kuala Lumpur together have 83.7% of the listings. The target classes are imbalanced (0: 2.7%, 1: 20.6%, 2: 26.9%, 3: 21.7%, 4: 28.0%).

## Most relevant features for prediction
Numerical and ordinal features were compared with the target by Spearman's rank correlation (ρ, rank-based, which fits the ordered target; *Not Specified* excluded). Nominal features were compared by Cramér's V (categories under 1% merged, so that rare breeds do not inflate V).

| Feature | Measure | Value | Observation |
| :---------- | :---- | ----: | :---------------------------------------------- |
| `Age` | ρ | 0.21 | Fastest at 0–2 months (mean AdoptionSpeed 2.24); 2.7–2.9 from 7 months on |
| `Sterilized` | V | 0.16 | Strongest nominal feature; probably partly an age effect |
| `Type` | V | 0.10 | Cats are adopted faster than dogs (mean 2.40 vs 2.62) |
| `Breed1` | V | 0.10 | 12 breeds with ≥ 1% of listings, the rest merged |
| `Vaccinated` | V | 0.10 | *Not Sure* treated as its own category |
| `FurLength` | ρ | −0.08 | Longer fur → slightly faster |
| `PhotoAmt` | ρ | −0.06 | More photos → slightly faster |

All associations are weak (maximum 0.21), so the model has to combine many weak signals. They show association, not causation, and ρ and V are different statistics, so comparing them with each other is only indicative.

## Data quality assessment
We checked missing values, code validity (against the description and the label files), cross-column consistency, identifiers, duplicates and outliers. Each finding was either **fixed** by `clean_petfinder()` or **kept** with a reason. **No rows were filtered**: none of the problems makes a row unusable.

| Finding | Rows (% of train) | Decision | Reason |
| :-------------------------- | ----------: | :-------- | :---------------------------- |
| `Name` missing (NaN) | 1,265 (8.4%) | keep | "Not named"; `has_name` flag in M2 |
| Placeholder name ("No Name", "Unknown" …) | 151 (1.0%) | fix → NaN | Means "no name" |
| `Description` missing (NaN) | 13 (0.1%) | fix → `""` | No NaN in text processing |
| *Not Sure* in `Vaccinated` / `Dewormed` / `Sterilized` | 1,868 / 1,781 / 1,815 (≈12%) | keep | Hidden missing value: own category, never used as a number |
| `Breed1 = 0` (the only invalid code) | 5 (0.03%) | fix | `Breed2` moved into `Breed1` |
| `Breed2` equals `Breed1` | 1,510 (10.1%) | fix → 0 | Redundant |
| Cat with a dog breed | 12 (0.1%) | keep | Unclear which field is wrong |
| Group listing (`Quantity > 1`) | 3,428 (22.9%) | keep | Legitimate; values describe the group |
| Identical rows apart from `PetID` | 16 (0.1%) | keep | Same rescuer and text; may be litter-mates |
| `Age` > 20 years / `Fee` > 1000 | 2 / 2 | keep | Suspicious but possible |
| `PhotoAmt` stored as float | all | fix → int | It is a count |

The following checks found nothing: duplicate `PetID`, *Not Specified* codes, colour gaps or repeats, and *Mixed* gender for a single pet.

**Missing values.** Only `Name` and `Description` contain NaN. The *Not Sure* answers are hidden missing values that `isna()` does not detect. Counting both kinds per row, 24.5% of the listings have at least one unknown value and 7.1% have three or more. If the answers were independent, fewer than 1% of rows would have three or more, so the unknowns cluster in the same listings, most likely those whose rescuer does not know the pet's history.

**Identifiers and rescuers.** `PetID` is unique, and no ID appears in both train and test. 5,595 rescuers posted the listings (median 1, maximum 459 each). **No training rescuer appears in the test set**, so the organisers split the data by rescuer. M2 validation must do the same (`GroupKFold` on `RescuerID`), otherwise scores are too optimistic.

**Outliers and distributions.**

| | `Age` (months) | `Quantity` | `Fee` | `VideoAmt` | `PhotoAmt` |
| :---------- | ----: | ----: | ----: | ----: | ----: |
| Median / Q3 / max | 3 / 12 / 255 | 1 / 1 / 20 | 0 / 0 / 3000 | 0 / 0 / 8 | 3 / 5 / 30 |
| Above IQR fence | 10.0% | 22.9% | 15.5% | 3.8% | 6.2% |

The IQR rule misleads for `Quantity`, `Fee` and `VideoAmt`: there Q1 = Q3, so every value above the most common one is flagged. `Age` is right-skewed and rounded: 74.6% of the ages from 12 months up are whole years, against 8.3% expected, so rescuers estimate adult ages. **No outliers were removed**, because the extreme values are possible. Skewness will be handled in M2 by transformation (`log1p`, capping).

**Reproducibility.** `clean_petfinder()` applies every fix to train and test in the same way. `assert` checks confirm that no rows were removed, the target is unchanged and the fixed problems are gone.
