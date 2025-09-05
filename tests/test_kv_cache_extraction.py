# tests/test_kv_cache_extraction.py

import os
import logging
import pytest
import pandas as pd

from slimsc.prune.evaluation.kv_cache_extraction import (
    clear_source_kv_cache,
    extract_kv_cache_usage_for_question,
)

# Sample CSV data for reuse in tests
SAMPLE_KV_CACHE_LOG = """1700000001.0,10.5
1700000002.0,20.0
invalid_timestamp,30.0
1700000004.0,40.0
1700000005.0,50.0
1700000006.0,not_a_number
1700000008.0,80.0
"""


@pytest.mark.unit
class TestClearSourceKVCache:
    """Tests for the clear_source_kv_cache function."""

    def test_clears_existing_file(self, tmp_path):
        """Verify that an existing file is successfully removed."""
        source_file = tmp_path / "kv_cache.log"
        source_file.touch()  # Create the file
        assert source_file.exists()

        clear_source_kv_cache(str(source_file))

        assert not source_file.exists()

    def test_handles_non_existent_file(self, tmp_path, caplog):
        """Verify it logs a warning but doesn't fail if the file doesn't exist."""
        source_file = tmp_path / "non_existent.log"
        assert not source_file.exists()

        with caplog.at_level(logging.WARNING):
            clear_source_kv_cache(str(source_file))
        
        assert f"KV cache usage file {source_file} does not exist" in caplog.text

    def test_handles_none_or_empty_path(self, caplog):
        """Verify it logs a warning if the file path is None or empty."""
        with caplog.at_level(logging.WARNING):
            clear_source_kv_cache(None)
        assert "Source KV cache file path not found" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            clear_source_kv_cache("")
        assert "Source KV cache file path not found" in caplog.text

    def test_logs_os_error_on_remove(self, tmp_path, monkeypatch, caplog):
        """Verify an OSError during removal is caught and logged."""
        source_file = tmp_path / "protected.log"
        source_file.touch()

        # Mock os.remove to raise an OSError
        def mock_remove(path):
            raise OSError("Permission denied")

        monkeypatch.setattr(os, "remove", mock_remove)

        with caplog.at_level(logging.ERROR):
            clear_source_kv_cache(str(source_file))

        assert f"Could not remove KV cache file {source_file}" in caplog.text
        assert "Permission denied" in caplog.text


@pytest.mark.unit
class TestExtractKVCacheUsage:
    """Tests for the extract_kv_cache_usage_for_question function."""

    @pytest.fixture
    def setup_paths(self, tmp_path):
        """Fixture to create source file and directories for a test."""
        source_dir = tmp_path / "source"
        save_dir = tmp_path / "saved"
        source_dir.mkdir()
        save_dir.mkdir()

        source_file = source_dir / "kv_cache.log"
        source_file.write_text(SAMPLE_KV_CACHE_LOG)
        
        paths = {
            'source_usage_file': str(source_file),
            'kvcache_usages_dir': str(save_dir)
        }
        return paths

    def test_happy_path_extraction_and_save(self, setup_paths):
        """Test standard extraction, calculation, and saving of filtered data."""
        paths = setup_paths
        start_time = 1700000003.0
        end_time = 1700000007.0
        iteration = 1

        results = extract_kv_cache_usage_for_question(start_time, end_time, iteration, paths)

        # Expected avg = (40.0 + 50.0) / 2 = 45.0
        # Expected max = 50.0
        assert results['avg_kv_cache_usage'] == pytest.approx(45.0)
        assert results['max_kv_cache_usage'] == pytest.approx(50.0)

        # Verify the saved file
        saved_file = os.path.join(paths['kvcache_usages_dir'], "question_1_kvcache_usage.csv")
        assert os.path.exists(saved_file)
        
        df_saved = pd.read_csv(saved_file)
        assert len(df_saved) == 2
        assert df_saved['timestamp'].tolist() == [1700000004.0, 1700000005.0]
        assert df_saved['gpu_cache_usage_perc'].tolist() == [40.0, 50.0]

    def test_no_data_in_time_window(self, setup_paths, caplog):
        """Test case where no log entries fall within the time window."""
        paths = setup_paths
        start_time = 1600000000.0  # Before all data
        end_time = 1600000010.0
        iteration = 2

        with caplog.at_level(logging.WARNING):
            results = extract_kv_cache_usage_for_question(start_time, end_time, iteration, paths)

        assert results['avg_kv_cache_usage'] is None
        assert results['max_kv_cache_usage'] is None
        assert "No KV cache usage data found within the time window" in caplog.text

        # Verify an empty file with a header was saved
        saved_file = os.path.join(paths['kvcache_usages_dir'], "question_2_kvcache_usage.csv")
        assert os.path.exists(saved_file)
        df_saved = pd.read_csv(saved_file)
        assert df_saved.empty
        assert list(df_saved.columns) == ['timestamp', 'gpu_cache_usage_perc']
        
    def test_source_file_not_found(self, tmp_path, caplog):
        """Test behavior when the source file does not exist."""
        paths = {
            'source_usage_file': str(tmp_path / 'non_existent.log'),
            'kvcache_usages_dir': str(tmp_path)
        }
        
        with caplog.at_level(logging.WARNING):
            results = extract_kv_cache_usage_for_question(1, 2, 1, paths)
            
        assert results['avg_kv_cache_usage'] is None
        assert results['max_kv_cache_usage'] is None
        assert f"KV cache usage file not found at {paths['source_usage_file']}" in caplog.text

    def test_source_path_not_configured(self, caplog):
        """Test behavior when 'source_usage_file' is missing from paths."""
        paths = {'kvcache_usages_dir': '/fake/dir'}
        with caplog.at_level(logging.WARNING):
            results = extract_kv_cache_usage_for_question(1, 2, 1, paths)

        assert results['avg_kv_cache_usage'] is None
        assert results['max_kv_cache_usage'] is None
        assert "Source KV cache usage file path not configured" in caplog.text

    def test_save_dir_not_configured(self, setup_paths, caplog):
        """Test that extraction works but saving is skipped if save dir is not configured."""
        paths = setup_paths
        paths.pop('kvcache_usages_dir')  # Remove save directory path
        
        start_time = 1700000003.0
        end_time = 1700000007.0
        iteration = 3
        
        with caplog.at_level(logging.WARNING):
            results = extract_kv_cache_usage_for_question(start_time, end_time, iteration, paths)

        # Calculations should still be correct
        assert results['avg_kv_cache_usage'] == pytest.approx(45.0)
        assert results['max_kv_cache_usage'] == pytest.approx(50.0)
        
        # Warning about not saving should be present
        assert "Save directory not configured. Skipping save" in caplog.text

    def test_empty_source_file(self, tmp_path, caplog):
        """Test behavior with an existing but empty source file."""
        source_file = tmp_path / "empty.log"
        source_file.touch()
        save_dir = tmp_path / "saved"
        save_dir.mkdir()
        
        paths = {
            'source_usage_file': str(source_file),
            'kvcache_usages_dir': str(save_dir)
        }
        
        with caplog.at_level(logging.WARNING):
            results = extract_kv_cache_usage_for_question(1, 2, 4, paths)

        assert results['avg_kv_cache_usage'] is None
        assert results['max_kv_cache_usage'] is None
        assert f"KV cache usage file {source_file} is empty" in caplog.text
        
        # Check that an empty results file was still created
        saved_file = save_dir / "question_4_kvcache_usage.csv"
        assert saved_file.exists()
        assert saved_file.read_text() == "timestamp,gpu_cache_usage_perc\n"
        
    def test_handles_bad_data_gracefully(self, setup_paths):
        """Test that non-numeric and malformed lines are ignored."""
        paths = setup_paths
        # Use a wide time window to include all valid data
        start_time = 1700000000.0
        end_time = 1700000010.0
        iteration = 5

        results = extract_kv_cache_usage_for_question(start_time, end_time, iteration, paths)

        # Expected data points: (10.5, 20.0, 40.0, 50.0, 80.0)
        # Avg = (10.5 + 20 + 40 + 50 + 80) / 5 = 200.5 / 5 = 40.1
        # Max = 80.0
        assert results['avg_kv_cache_usage'] == pytest.approx(40.1)
        assert results['max_kv_cache_usage'] == pytest.approx(80.0)