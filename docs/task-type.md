# PetFinder task type

## Target variable
The prediction target is `AdoptionSpeed`, which represents how quickly a pet is adopted after listing (lower means faster adoption).

## Target values
- `0`: adopted on the same day
- `1`: adopted in 1-7 days
- `2`: adopted in 8-30 days
- `3`: adopted in 31-90 days
- `4`: not adopted after 100 days

## Task type
This is a **multiclass classification** task, not regression.

Reason: the model predicts one of five discrete category labels (`0-4`) that represent adoption-speed classes, not a continuous numeric value. Even though labels are ordered, the objective is still to classify each pet into a category.

## Target encoding / transformation
No extra target transformation is applied in this documentation. The provided `AdoptionSpeed` integer labels (`0-4`) are used directly as categorical class IDs.
