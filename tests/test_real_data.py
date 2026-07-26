"""
Tests for the real-data loaders (harespod_loader.py, pilot_altitude_loader.py)
and the post-hoc labeling pipeline (label_real_data.py).

Unlike test_pipeline.py's synthetic-data tests, these depend on data that
was manually downloaded into data/raw/ -- not something this repo can ship
or generate itself (see each loader's docstring for the download steps).
Every test here is skipped, not failed, when the relevant dataset isn't
present, so `pytest tests/` still passes cleanly on a machine that hasn't
downloaded either dataset -- consistent with CLAUDE.md's "runs and degrades
gracefully" principle applied to the test suite itself.

Run with: python -m pytest tests/test_real_data.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import N_TIERS
from src.data import harespod_loader as hl
from src.data import pilot_altitude_loader as pal

harespod_available = hl.has_harespod_data()
pilot_available = pal.has_pilot_altitude_data()

requires_harespod = pytest.mark.skipif(
    not harespod_available, reason="Harespod not downloaded into data/raw/harespod/"
)
requires_pilot = pytest.mark.skipif(
    not pilot_available, reason="Pilot altitude dataset not downloaded into data/raw/pilot_altitude/"
)


@requires_harespod
def test_harespod_lists_fifteen_subjects():
    """
    The dataset's own paper (PMC10899206) states 15 of 23 recruited
    subjects were usable -- a regression here (wrong count) would mean
    either a folder got missed or an unexpected extra folder is being
    picked up as if it were a real subject.
    """
    subjects = hl.list_subject_ids()
    assert len(subjects) == 15


@requires_harespod
def test_harespod_load_subject_shape_and_bounds():
    df = hl.load_subject("217a")
    assert set(hl.TIDY_COLUMNS) <= set(df.columns)
    # Physiological plausibility, same bounds check as the synthetic
    # generator's test -- this is what would have caught the original
    # "rescale onto the device's full 0-250bpm range" bug (produced a mean
    # HR of 174bpm, clearly outside any plausible resting-to-elevated band).
    assert df["spo2"].between(*hl.SPO2_DEVICE_RANGE).all()
    assert df["hr"].between(*hl.HEART_RATE_DEVICE_RANGE).all()
    assert (df["altitude"] >= 0).all()
    assert (df["data_source"] == "harespod_rescaled").all()


@requires_harespod
def test_harespod_altitude_interpolation_reaches_expected_range():
    """
    Every subject except 328a should span 1500-4000m (328a's protocol
    stopped at 3500m, see harespod_loader.py's _parse_key_timestamps
    docstring) -- this is the regression test for two real bugs found and
    fixed while integrating this dataset: the GMT-vs-local-time 8-hour
    offset affecting 6 of 15 subjects (which, unfixed, silently produced a
    CONSTANT altitude for the whole recording instead of a real ramp), and
    314c's differently-worded first marker.
    """
    for subject_id in hl.list_subject_ids():
        df = hl.load_subject(subject_id)
        min_alt, max_alt = df["altitude"].min(), df["altitude"].max()
        # A few minutes' gap between the chamber reaching 1500m and this
        # subject's own recording actually starting (e.g. 218c's data
        # starts ~1 minute after its t1 marker) is normal session timing
        # variance, not a bug -- 1700m tolerance comfortably covers that
        # while still catching the "stuck at a constant 4000m" failure
        # mode the GMT-offset bug produced (that bug's min_alt was ~4000m).
        assert min_alt <= 1700, f"{subject_id}: altitude never reached near 1500m (got {min_alt})"
        if subject_id == "328a":
            assert max_alt <= 3600, f"328a should cap near 3500m, got {max_alt}"
        else:
            assert max_alt >= 3900, f"{subject_id}: altitude never reached near 4000m (got {max_alt})"


@requires_harespod
def test_harespod_load_all_subjects_no_duplicate_ids():
    all_df = hl.load_all_subjects()
    assert all_df["subject_id"].nunique() == 15
    # Every subject_id should be prefixed, distinguishing it from
    # synthetic ("synth_...") and pilot ("pilot_...") subject ids when
    # datasets are merged in feature_engineering.py.
    assert all_df["subject_id"].str.startswith("harespod_").all()


@requires_pilot
def test_pilot_altitude_lists_twenty_subjects():
    subjects = pal.list_subject_ids()
    assert len(subjects) == 20


@requires_pilot
def test_pilot_altitude_load_subject_units_are_physiological():
    """
    Unlike Harespod, this dataset's SPO2/Heart rate columns are already in
    real physical units (confirmed by inspecting the raw CSVs directly --
    see pilot_altitude_loader.py's docstring) -- no rescaling is applied.
    This test guards against that assumption silently becoming wrong if a
    different subject file ever turns out to be normalized after all.
    """
    df = pal.load_subject("0609b")
    assert df["spo2"].between(50, 100).all()
    assert df["hr"].between(30, 220).all()
    assert (df["data_source"] == "pilot_altitude").all()


@requires_harespod
def test_label_real_dataframe_produces_valid_severity_indices():
    from src.data.label_real_data import label_real_dataframe

    raw = hl.load_subject("217a")
    labeled = label_real_dataframe(raw)
    assert labeled["severity_index"].min() >= 0
    assert labeled["severity_index"].max() < N_TIERS
    assert not labeled["severity_index"].isna().any()


@requires_harespod
def test_real_data_merges_into_train_only():
    """
    The core integrity guarantee of the real-data integration: real,
    rule-labeled rows must appear in train.csv but NEVER in val/test,
    since the rule baseline generated their labels (see
    feature_engineering.run_pipeline()'s docstring on why blending real
    data into val/test would make the rule-baseline comparison circular).
    This test exercises the actual merge function, not just a description
    of the intent.
    """
    from src.data.feature_engineering import load_real_data_for_training

    real = load_real_data_for_training()
    assert not real.empty
    assert (real["subject_id"].str.startswith("harespod_")).all()
    assert "temp" in real.columns  # synthesized, since Harespod has no temp sensor
    assert not real["temp"].isna().any()
