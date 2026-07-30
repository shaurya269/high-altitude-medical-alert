"""
DataSource interface package -- see base.py for the abstract contract
every concrete source (scenario player, manual override, Harespod replay,
and a future arduino_reader.py) implements.
"""

from src.datasource.base import DataSource, Reading
from src.datasource.buffer import ReadingBuffer
from src.datasource.harespod_replay import HarespodReplayDataSource
from src.datasource.harespod_upload import HarespodUploadDataSource, load_uploaded_subject
from src.datasource.manual_override import ManualDataSource
from src.datasource.scenario_player import SCENARIOS, ScenarioDataSource

__all__ = [
    "DataSource",
    "Reading",
    "ReadingBuffer",
    "ScenarioDataSource",
    "SCENARIOS",
    "ManualDataSource",
    "HarespodReplayDataSource",
    "HarespodUploadDataSource",
    "load_uploaded_subject",
]
