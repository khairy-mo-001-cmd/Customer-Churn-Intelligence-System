from pathlib import Path

import pytest

from src.ingestion.download_raw import load_raw_transactions


def test_load_csv_parquet_and_multi_sheet_xlsx(tmp_path, raw_dirty):
    raw_dirty.to_csv(tmp_path / 'a.CSV', index=False)  # upper-case suffix is accepted
    assert len(load_raw_transactions(tmp_path / 'a.CSV')) == len(raw_dirty)

    raw_dirty.to_parquet(tmp_path / 'a.parquet')
    assert len(load_raw_transactions(tmp_path / 'a.parquet')) == len(raw_dirty)

    import pandas as pd
    with pd.ExcelWriter(tmp_path / 'a.xlsx') as writer:
        raw_dirty.to_excel(writer, sheet_name='Year 1', index=False)
        raw_dirty.to_excel(writer, sheet_name='Year 2', index=False)
    assert len(load_raw_transactions(tmp_path / 'a.xlsx')) == 2 * len(raw_dirty)


def test_missing_file_and_bad_format(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_transactions(Path('nonexistent.csv'))
    bad = tmp_path / 'x.txt'
    bad.write_text('dummy')
    with pytest.raises(ValueError, match='Unsupported file format'):
        load_raw_transactions(bad)
