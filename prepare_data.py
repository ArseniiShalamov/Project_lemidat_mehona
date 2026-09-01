from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
TEST_FRACTION = 0.15


def stratified_train_test_split(df, label_col='label', test_fraction=0.15, seed=42):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for label, group in df.groupby(label_col, sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_parts.append(df.loc[test_idx])
        train_parts.append(df.loc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)
    test_df = test_df.iloc[rng.permutation(len(test_df))].reset_index(drop=True)
    return train_df, test_df


def main():
    # Preferred source: the Kaggle file already downloaded into the project directory.
    candidates = [
        Path('original_train.csv'),
        Path('kaggle_train.csv'),
        Path('source_train.csv'),
        Path('train_original.csv'),
    ]
    source = next((p for p in candidates if p.exists()), None)

    if source is None:
        try:
            import kagglehub
            dataset_dir = Path(kagglehub.dataset_download(
                'thedevastator/sms-spam-collection-a-more-diverse-dataset'
            ))
            csv_files = sorted(dataset_dir.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError('No CSV file was found in the Kaggle download.')
            source = csv_files[0]
            print('Downloaded source:', source)
        except Exception as exc:
            raise SystemExit(
                'Could not find a local source CSV and Kaggle download failed.\n'
                'Install kagglehub with: pip install kagglehub\n'
                f'Details: {exc}'
            )

    df = pd.read_csv(source)
    if not {'sms', 'label'}.issubset(df.columns):
        raise ValueError(f'Expected columns sms,label. Found: {df.columns.tolist()}')

    df = df[['sms', 'label']].copy()
    df['sms'] = df['sms'].astype(str)
    df['label'] = df['label'].astype(int)

    # Exact-message deduplication before splitting prevents the same SMS from
    # appearing in both train and test. Whitespace is ignored only for the key.
    df['_dedupe_key'] = df['sms'].str.strip()

    conflicts = df.groupby('_dedupe_key')['label'].nunique()
    conflicting_keys = conflicts[conflicts > 1].index
    if len(conflicting_keys):
        raise ValueError(
            f'Found {len(conflicting_keys)} SMS texts with conflicting labels. '
            'Resolve them before splitting.'
        )

    before = len(df)
    df = df.drop_duplicates(subset='_dedupe_key', keep='first').drop(columns='_dedupe_key')
    removed = before - len(df)

    train_df, test_df = stratified_train_test_split(
        df,
        label_col='label',
        test_fraction=TEST_FRACTION,
        seed=RANDOM_SEED,
    )

    train_keys = set(train_df['sms'].str.strip())
    test_keys = set(test_df['sms'].str.strip())
    assert train_keys.isdisjoint(test_keys)

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print(f'Original rows: {before}')
    print(f'Duplicate SMS rows removed: {removed}')
    print(f'Rows after deduplication: {len(df)}')
    print(f'Train shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')
    print('\nTrain labels:')
    print(train_df['label'].value_counts().sort_index())
    print('\nTest labels:')
    print(test_df['label'].value_counts().sort_index())
    print('\nNo train/test SMS overlap:', train_keys.isdisjoint(test_keys))


if __name__ == '__main__':
    main()
