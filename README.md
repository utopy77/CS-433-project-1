# CS-433 Machine Learning — Project 1

Predicting the presence of coronary heart disease or myocardial infarction (`MICHD`) from the [CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html) health survey dataset, implemented from scratch using only **NumPy**.

For a full discussion of the exploratory data analysis, domain-driven preprocessing, cross-validation protocol, model comparisons, and ablation study, please refer to our [`project report`](project_1_report.pdf) and [`project1_notebook.ipynb`](project1_notebook.ipynb).

---

## Repository Structure

```text
├── dataset/                    # Training and test CSV files (x_train.csv, y_train.csv, x_test.csv)
├── grading_tests/              # Public test suite for checking function signatures and behavior
├── helpers.py                  # Data loading and AIcrowd CSV submission utilities
├── implementations.py          # Core ML algorithms (GD, SGD, Least Squares, Ridge, Logistic, Reg. Logistic)
├── preprocessing.py            # BRFSS special-code recoding, imputation, one-hot encoding, standardization
├── validation.py               # Metrics (F1, Accuracy, AUC), stratified k-fold CV, and threshold tuning
├── run.py                      # End-to-end script reproducing the final submission.csv
├── Data_Exploration.ipynb      # Initial exploratory data analysis
└── project1_notebook.ipynb     # Full experimental pipeline, cross-validation results, and ablations
```

---

## Setup & Requirements

The core pipeline (`run.py`, `implementations.py`, `preprocessing.py`, `validation.py`, `helpers.py`) only requires **Python $\ge$ 3.9** and **NumPy**:

```bash
pip install numpy
```

To run the notebooks or the grading test suite, you can create the provided Conda environment:

```bash
conda env create --file=grading_tests/environment.yml --name=project1-grading
conda activate project1-grading
```

### Dataset

Place the competition data files in the `dataset/` directory at the root of the repository:

```text
dataset/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

---

## Reproducing the Final Submission

To train the final model and generate `submission.csv` at the root of the repository, run:

```bash
python run.py
```

`run.py` executes the following steps:
1. **Recoding & Preprocessing (`preprocessing.py`):** Excludes administrative, redundant, and leakage columns; recodes BRFSS questionnaire special codes (`77`/`88`/`99`, etc.) to `NaN` or `0`; drops features with $> 80\%$ missing values ($\tau = 0.8$); applies median imputation, missingness indicators, and one-hot encoding of nominal variables; and standardizes all features using training statistics only.
2. **Threshold Selection:** Uses a stratified $4/5$–$1/5$ inner split of the training set (`SEED = 42`) to select the probability threshold that maximizes the $F_1$ score under class imbalance ($p \approx 8.8\%$).
3. **Model Training (`implementations.py`):** Fits unregularized logistic regression on the full training set via gradient descent ($1000$ iterations) with adaptive Lipschitz step size $\gamma = 1 / L$, where $L = \lambda_{\max}(X^\top X / n) / 4$.
4. **Prediction:** Outputs binary predictions in $\{-1, 1\}$ to `submission.csv` (achieving **$F_1 = 0.439$** and **$87.5\%$ accuracy** on the AIcrowd test leaderboard).

---

## Running the Grading Tests

To verify the implementations in `implementations.py` against the public unit tests:

```bash
pytest grading_tests/test_project1_public.py
```
