# ML NLP KNN Spam Project

Binary SMS spam classification for the Machine Learning NLP/Text Analysis assignment.

## Students

- Arsenii S — ID ending `2319`
- Kristina S — ID ending `7553`
- Maxim G — ID ending `7344`

## Dataset

Kaggle dataset: **SMS Spam Collection (Text Classification)**  
https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset?resource=download

The original Kaggle file contains 5,574 labeled SMS messages. Before the fixed project split, exact duplicate SMS texts are removed (leading/trailing whitespace is ignored only for duplicate detection). This prevents the same SMS from appearing in both train and test.

The deduplicated data is split once with a fixed stratified random seed (`42`) into:

- `train.csv` — 4,386 messages
- `test.csv` — 774 messages

The notebook does not split the test set again. Model selection is performed only on the training set using stratified 5-Fold Cross Validation.

## Files

- `spam_classification_knn.ipynb` — main notebook, already executed with saved outputs
- `train.csv` — fixed training set
- `test.csv` — fixed untouched test set
- `prepare_data.py` — reproducible data-preparation script using `kagglehub` if the original Kaggle CSV is not present locally
- `requirements.txt` — Python dependencies

## Assignment flow

1. Load fixed train and test data and show the first 5 rows.
2. Verify class distribution and train/test separation.
3. Use F1-score for the central spam class.
4. Implement text preprocessing and TF-IDF from scratch.
5. Show Feature Engineering on train and test examples.
6. Implement KNN from scratch with `fit`/`predict`, `k`, and uniform/weighted voting.
7. Run a Cartesian Grid Search over Feature Engineering and KNN settings.
8. Wrap all grid permutations in stratified 5-Fold Cross Validation.
9. Compare random oversampling for imbalanced data using Cross Validation.
10. Retrain the selected configuration on the complete train set.
11. Evaluate once on the fixed test set and show the first 5 predictions, F1, Precision, Recall, Accuracy and confusion matrix.
12. Explain a KNN prediction using its nearest neighbors, cosine similarity/distance and vote weights.

## Bonus experiments

### Bonus 6A — Grid Search + 5-Fold Cross Validation

The grid includes:

- Feature Engineering: `max_features`, `min_df`, unigram vs unigram+bigrams
- KNN hyperparameters: `k`, uniform vs weighted voting

All Cartesian-product permutations and their mean 5-Fold spam F1 are displayed in a DataFrame, followed by the best configuration separately.

### Bonus 6B — Imbalanced data

Random oversampling is evaluated inside 5-Fold Cross Validation. Oversampling is applied only to training folds, never to validation/test examples.

### Bonus 6C — Explainability

A selected test prediction is explained using the nearest training examples, cosine similarity/distance, labels and vote weights.

## Final saved run

The notebook was run from a clean kernel and its outputs are saved. In the included fixed split, the final saved run selected:

- `max_features = 600`
- `min_df = 1`
- `ngram_mode = unigram`
- `k = 5`
- `voting = weighted`
- mean 5-Fold spam F1 ≈ `0.8996`

Final test spam F1 ≈ `0.8977`.

The assignment emphasizes the learning process and explanation, not only the final model score.

## Reproducing the data

The project already contains the fixed `train.csv` and `test.csv`, so this step is not required for submission.

To reproduce them from Kaggle:

```bash
pip install -r requirements.txt
python3 prepare_data.py
```

`prepare_data.py` uses:

```python
import kagglehub
path = kagglehub.dataset_download(
    "thedevastator/sms-spam-collection-a-more-diverse-dataset"
)
```

## Before submission

Open the notebook on GitHub/Colab and verify that all outputs and plots are visible. The ~5-minute presentation/video should show the code and outputs and demonstrate understanding; all group members should participate approximately equally.
