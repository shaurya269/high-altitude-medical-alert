"""
Tests for the Day 8 DataSource interface (src/datasource/): the scenario
player, manual override, Harespod replay, and the rolling buffer that sits
between any of them and feature engineering.

Run with: python -m pytest tests/test_datasource.py -v
"""

from __future__ import annotations

import pytest

from src.data import harespod_loader as hl
from src.data.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.datasource import (
    ManualDataSource,
    ReadingBuffer,
    ScenarioDataSource,
)
from src.datasource.base import Reading
from src.datasource.scenario_player import SCENARIOS

harespod_available = hl.has_harespod_data()
requires_harespod = pytest.mark.skipif(
    not harespod_available, reason="Harespod not downloaded into data/raw/harespod/"
)


def test_scenario_player_covers_all_five_tiers():
    """SCENARIOS must offer exactly the five CLAUDE.md Section 2 scenarios
    (Normal / Mild AMS / Severe AMS / HAPE onset / HACE risk) -- a
    regression here (a dropped or renamed key) would silently remove an
    option from the Demo Mode UI's scenario picker."""
    assert set(SCENARIOS) == {"Normal", "Mild AMS", "Severe AMS", "HAPE onset", "HACE risk"}


def test_scenario_player_reading_shape():
    source = ScenarioDataSource("Mild AMS", duration_minutes=5, seed=1)
    reading = source.next_reading()
    assert set(reading.keys()) == {"timestamp", "spo2", "hr", "temp", "altitude"}


def test_scenario_player_deterministic_with_seed():
    """Same seed must produce the same trajectory -- this is what lets a
    Streamlit rerun re-render the same scenario without the data silently
    changing underneath the user (see scenario_player.py's docstring)."""
    a = ScenarioDataSource("Severe AMS", duration_minutes=5, seed=99)
    b = ScenarioDataSource("Severe AMS", duration_minutes=5, seed=99)
    readings_a = [a.next_reading() for _ in range(10)]
    readings_b = [b.next_reading() for _ in range(10)]
    assert readings_a == readings_b


def test_scenario_player_exhausts_and_resets():
    source = ScenarioDataSource("Normal", duration_minutes=1, seed=1)
    n = len(source)
    for _ in range(n):
        assert source.next_reading() is not None
    assert source.next_reading() is None  # exhausted
    source.reset()
    assert source.next_reading() is not None  # available again after reset


def test_scenario_player_unknown_name_raises():
    with pytest.raises(ValueError):
        ScenarioDataSource("Not A Real Scenario")


def test_manual_override_holds_values_until_changed():
    source = ManualDataSource()
    r1 = source.next_reading()
    r2 = source.next_reading()
    assert r1["spo2"] == r2["spo2"] == r1["hr"] == r2["hr"] or (r1["spo2"] == r2["spo2"])
    assert r2["timestamp"] > r1["timestamp"]  # timestamp still advances

    source.set_values(spo2=80.0)
    r3 = source.next_reading()
    assert r3["spo2"] == 80.0
    assert r3["hr"] == r1["hr"]  # unset fields stay unchanged


def test_manual_override_reset_restores_defaults():
    source = ManualDataSource()
    source.set_values(spo2=70.0, hr=140.0)
    source.reset()
    reading = source.next_reading()
    assert reading["spo2"] != 70.0  # back to default, not the overridden value


def test_reading_buffer_rejects_out_of_range_values():
    buffer = ReadingBuffer()
    good: Reading = {"timestamp": 0.0, "spo2": 95.0, "hr": 75.0, "temp": 36.8, "altitude": 2000.0}
    glitch: Reading = {"timestamp": 1.0, "spo2": -5.0, "hr": 75.0, "temp": 36.8, "altitude": 2000.0}

    assert buffer.add(good) is True
    assert buffer.add(glitch) is False
    assert len(buffer) == 1
    assert buffer.rejected_count == 1


def test_reading_buffer_is_rolling_not_unbounded():
    """The buffer must cap at ROLLING_BUFFER_MINUTES worth of readings,
    not grow forever -- this is Stage 2's actual "rolling" behavior, and a
    regression here (e.g. an unbounded list instead of deque(maxlen=...))
    would silently blow up memory during a long live session."""
    from src.config import ROLLING_BUFFER_MINUTES, SAMPLE_RATE_HZ

    maxlen = ROLLING_BUFFER_MINUTES * 60 * SAMPLE_RATE_HZ
    buffer = ReadingBuffer()
    for i in range(maxlen + 100):
        buffer.add({"timestamp": float(i), "spo2": 95.0, "hr": 75.0, "temp": 36.8, "altitude": 2000.0})
    assert len(buffer) == maxlen


def test_reading_buffer_as_dataframe_feeds_engineer_features():
    """End-to-end smoke test: a DataSource -> buffer -> engineer_features
    chain must produce every column FEATURE_COLUMNS expects, since that's
    the exact chain the live pipeline (and the Streamlit app) will run."""
    source = ScenarioDataSource("Mild AMS", duration_minutes=8, seed=5)
    buffer = ReadingBuffer()
    for _ in range(len(source)):
        reading = source.next_reading()
        if reading is None:
            break
        buffer.add(reading)

    featured = engineer_features(buffer.as_dataframe())
    for col in FEATURE_COLUMNS:
        assert col in featured.columns
    assert not featured[FEATURE_COLUMNS].isna().any().any()


@requires_harespod
def test_harespod_replay_matches_available_subjects():
    from src.datasource import HarespodReplayDataSource

    subjects = HarespodReplayDataSource.available_subjects()
    assert len(subjects) == 15

    source = HarespodReplayDataSource(subjects[0], seed=1)
    reading = source.next_reading()
    assert set(reading.keys()) == {"timestamp", "spo2", "hr", "temp", "altitude"}


@requires_harespod
def test_uploaded_subject_matches_the_real_loader_for_the_same_subject():
    """The upload path (harespod_upload.py) re-derives spo2/hr/altitude via
    the SAME rescale/interpolation helpers harespod_loader.py's own
    load_subject() uses -- reading the identical 3 files for a real subject
    through both paths must produce identical numbers, or the upload
    adapter has silently drifted from the tested loader."""
    from src.data import harespod_loader as hl
    from src.datasource.harespod_upload import load_uploaded_subject

    subject_id = hl.list_subject_ids()[0]
    subject_dir = hl.HARESPOD_DIR / "Data_Cons" / subject_id
    uploaded_files = {
        "hr_5cut.csv": (subject_dir / "hr_5cut.csv").read_bytes(),
        "spv_5cut.csv": (subject_dir / "spv_5cut.csv").read_bytes(),
        "key_timestamp.txt": (subject_dir / "key_timestamp.txt").read_bytes(),
    }

    via_upload = load_uploaded_subject(uploaded_files)
    via_real_loader = hl.load_subject(subject_id)

    assert len(via_upload) == len(via_real_loader)
    assert via_upload["spo2"].equals(via_real_loader["spo2"])
    assert via_upload["hr"].equals(via_real_loader["hr"])
    assert via_upload["altitude"].equals(via_real_loader["altitude"])
    # subject_id/data_source columns are the one deliberate difference --
    # the upload path always tags "harespod_uploaded", never the real
    # subject's own code, since an ad-hoc upload isn't one of the 15 known
    # subjects the rest of the pipeline might otherwise assume it is.
    assert via_upload["subject_id"].iloc[0] == "harespod_uploaded"


def test_uploaded_subject_missing_file_raises_a_clear_error():
    from src.datasource.harespod_upload import load_uploaded_subject

    with pytest.raises(ValueError, match="Missing required file"):
        load_uploaded_subject({"hr_5cut.csv": b"irrelevant"})


@requires_harespod
def test_uploaded_subject_replays_through_the_full_pipeline():
    """Confirms the upload path satisfies the DataSource contract well
    enough for MedicalAlertPipeline to run it end to end, same as any
    other source."""
    from src.data import harespod_loader as hl
    from src.datasource.harespod_upload import HarespodUploadDataSource, load_uploaded_subject
    from src.pipeline import MedicalAlertPipeline

    subject_id = hl.list_subject_ids()[0]
    subject_dir = hl.HARESPOD_DIR / "Data_Cons" / subject_id
    uploaded_files = {
        "hr_5cut.csv": (subject_dir / "hr_5cut.csv").read_bytes(),
        "spv_5cut.csv": (subject_dir / "spv_5cut.csv").read_bytes(),
        "key_timestamp.txt": (subject_dir / "key_timestamp.txt").read_bytes(),
    }
    df = load_uploaded_subject(uploaded_files)
    source = HarespodUploadDataSource(df, seed=1)
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=60)

    result = None
    for _ in range(70):
        result = pipeline.tick()
        if result.exhausted:
            break
    assert result is not None
    assert result.severity is not None
    assert result.severity["severity_label"] in [
        "Normal", "Mild AMS", "Severe AMS", "HAPE risk", "HACE risk",
    ]
