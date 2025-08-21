"""Tests for prune.utils.precompute module."""

import pytest
from unittest.mock import patch, Mock, MagicMock, mock_open
import json
import warnings
from pathlib import Path

from prune.utils.precompute import calculate_overall_avg_kv_cache_for_run, main


class TestCalculateOverallAvgKvCacheForRun:
    """Test the calculate_overall_avg_kv_cache_for_run function."""
    
    def test_calculate_with_valid_data(self, temp_dir, mock_json_file):
        """Test calculation with valid KV cache data."""
        # Create test data
        test_data = [
            {"avg_kv_cache_usage": 0.75, "other_key": "value1"},
            {"avg_kv_cache_usage": 0.85, "other_key": "value2"},
            {"avg_kv_cache_usage": 0.65, "other_key": "value3"}
        ]
        
        # Create summaries directory and files
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        json_files = []
        for i, data in enumerate(test_data):
            file_path = mock_json_file(data, f"question_{i}_summary.json")
            # Move to summaries directory
            new_path = summaries_dir / f"question_{i}_summary.json"
            file_path.rename(new_path)
            json_files.append(new_path)
        
        # Test the function
        result = calculate_overall_avg_kv_cache_for_run(temp_dir)
        
        # Should return the mean of [0.75, 0.85, 0.65] = 0.75
        expected_mean = (0.75 + 0.85 + 0.65) / 3
        assert abs(result - expected_mean) < 1e-10
    
    def test_calculate_no_summary_files(self, temp_dir):
        """Test when no summary files are found."""
        # Create empty summaries directory
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            assert result is None
            assert len(w) == 1
            assert "No summary JSON files found" in str(w[0].message)
    
    def test_calculate_no_summaries_directory(self, temp_dir):
        """Test when summaries directory doesn't exist."""
        # Don't create the summaries directory
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            assert result is None
            assert len(w) == 1
            assert "No summary JSON files found" in str(w[0].message)
    
    def test_calculate_missing_kv_cache_key(self, temp_dir, mock_json_file):
        """Test with JSON files missing avg_kv_cache_usage key."""
        test_data = [
            {"other_key": "value1"},  # Missing avg_kv_cache_usage
            {"avg_kv_cache_usage": 0.8, "other_key": "value2"},
            {"different_key": "value3"}  # Missing avg_kv_cache_usage
        ]
        
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        for i, data in enumerate(test_data):
            file_path = mock_json_file(data, f"question_{i}_summary.json")
            new_path = summaries_dir / f"question_{i}_summary.json"
            file_path.rename(new_path)
        
        result = calculate_overall_avg_kv_cache_for_run(temp_dir)
        
        # Should only use the one valid value: 0.8
        assert result == 0.8
    
    def test_calculate_invalid_kv_cache_types(self, temp_dir, mock_json_file):
        """Test with invalid types for avg_kv_cache_usage."""
        test_data = [
            {"avg_kv_cache_usage": "not_a_number"},
            {"avg_kv_cache_usage": 0.75},  # Valid
            {"avg_kv_cache_usage": None},
            {"avg_kv_cache_usage": []},  # Invalid type
            {"avg_kv_cache_usage": 0.85}  # Valid
        ]
        
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        for i, data in enumerate(test_data):
            file_path = mock_json_file(data, f"question_{i}_summary.json")
            new_path = summaries_dir / f"question_{i}_summary.json"
            file_path.rename(new_path)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            
            # Should only use valid values: 0.75, 0.85
            expected_mean = (0.75 + 0.85) / 2
            assert abs(result - expected_mean) < 1e-10
            
            # Should have warnings for invalid types
            warning_messages = [str(warning.message) for warning in w]
            assert any("Unexpected type" in msg for msg in warning_messages)
    
    def test_calculate_all_invalid_data(self, temp_dir, mock_json_file):
        """Test when all files have invalid or missing data."""
        test_data = [
            {"other_key": "value1"},
            {"avg_kv_cache_usage": "not_a_number"},
            {"avg_kv_cache_usage": None}
        ]
        
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        for i, data in enumerate(test_data):
            file_path = mock_json_file(data, f"question_{i}_summary.json")
            new_path = summaries_dir / f"question_{i}_summary.json"
            file_path.rename(new_path)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            
            assert result is None
            warning_messages = [str(warning.message) for warning in w]
            assert any("No valid 'avg_kv_cache_usage' data found" in msg for msg in warning_messages)
    
    def test_calculate_json_decode_error(self, temp_dir):
        """Test handling of JSON decode errors."""
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        # Create a file with invalid JSON
        invalid_json_file = summaries_dir / "question_0_summary.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content")
        
        # Create a valid file too
        valid_file = summaries_dir / "question_1_summary.json"
        with open(valid_file, 'w') as f:
            json.dump({"avg_kv_cache_usage": 0.8}, f)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            
            # Should use the valid file
            assert result == 0.8
            
            # Should have warning about JSON decode error
            warning_messages = [str(warning.message) for warning in w]
            assert any("Error decoding JSON" in msg for msg in warning_messages)
    
    def test_calculate_file_not_found_error(self, temp_dir):
        """Test handling when a file is deleted between glob and open."""
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        # Create a file that we'll mock to not exist when opened
        test_file = summaries_dir / "question_0_summary.json"
        with open(test_file, 'w') as f:
            json.dump({"avg_kv_cache_usage": 0.8}, f)
        
        # Mock the glob to return the file, but open to raise FileNotFoundError
        with patch('pathlib.Path.glob') as mock_glob, \
             warnings.catch_warnings(record=True) as w:
            
            warnings.simplefilter("always")
            mock_glob.return_value = [test_file]
            
            # Mock open to raise FileNotFoundError
            with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
                result = calculate_overall_avg_kv_cache_for_run(temp_dir)
                
                assert result is None
                warning_messages = [str(warning.message) for warning in w]
                assert any("Summary file not found during processing" in msg for msg in warning_messages)
    
    def test_calculate_unexpected_exception(self, temp_dir):
        """Test handling of unexpected exceptions."""
        summaries_dir = temp_dir / "summaries"
        summaries_dir.mkdir()
        
        test_file = summaries_dir / "question_0_summary.json"
        with open(test_file, 'w') as f:
            json.dump({"avg_kv_cache_usage": 0.8}, f)
        
        # Mock json.load to raise an unexpected exception
        with patch('json.load', side_effect=Exception("Unexpected error")), \
             warnings.catch_warnings(record=True) as w:
            
            warnings.simplefilter("always")
            result = calculate_overall_avg_kv_cache_for_run(temp_dir)
            
            assert result is None
            warning_messages = [str(warning.message) for warning in w]
            assert any("An unexpected error occurred processing" in msg for msg in warning_messages)


class TestMain:
    """Test the main function."""
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"])
    def test_main_basic_execution(self, mock_base_path, temp_dir, capsys):
        """Test basic execution of main function."""
        # Set up the mock to return the temp directory structure
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        # Create the directory structure
        model_dir = temp_dir / "TestModel"
        model_dir.mkdir()
        dataset_dir = model_dir / "test_dataset"
        dataset_dir.mkdir()
        
        # Create an sc_X_control directory
        sc_dir = dataset_dir / "sc_8_control"
        sc_dir.mkdir()
        
        # Mock calculate_overall_avg_kv_cache_for_run
        with patch('prune.utils.precompute.calculate_overall_avg_kv_cache_for_run') as mock_calc:
            mock_calc.return_value = 0.75
            
            # Mock the output file writing
            with patch('builtins.open', mock_open()) as mock_file:
                main()
                
                # Check that the function was called
                mock_calc.assert_called_once_with(sc_dir)
                
                # Check that output was written
                mock_file.assert_called()
                
                # Check console output
                captured = capsys.readouterr()
                assert "Starting precomputation" in captured.out
                assert "Successfully processed and saved: 1 directories" in captured.out
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"])
    def test_main_no_dataset_directory(self, mock_base_path, temp_dir, capsys):
        """Test main when dataset directory doesn't exist."""
        # Set up the mock to return the temp directory but don't create the structure
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        main()
        
        captured = capsys.readouterr()
        assert "INFO: Dataset path not found, skipping" in captured.out
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"])
    def test_main_no_sc_directories(self, mock_base_path, temp_dir, capsys):
        """Test main when no sc_X_control directories exist."""
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        # Create the directory structure but no sc_X_control directories
        model_dir = temp_dir / "TestModel"
        model_dir.mkdir()
        dataset_dir = model_dir / "test_dataset"
        dataset_dir.mkdir()
        
        main()
        
        captured = capsys.readouterr()
        assert "Successfully processed and saved: 0 directories" in captured.out
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"])
    def test_main_calculation_returns_none(self, mock_base_path, temp_dir, capsys):
        """Test main when calculation returns None."""
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        # Create the directory structure
        model_dir = temp_dir / "TestModel"
        model_dir.mkdir()
        dataset_dir = model_dir / "test_dataset"
        dataset_dir.mkdir()
        sc_dir = dataset_dir / "sc_8_control"
        sc_dir.mkdir()
        
        # Mock calculate_overall_avg_kv_cache_for_run to return None
        with patch('prune.utils.precompute.calculate_overall_avg_kv_cache_for_run') as mock_calc:
            mock_calc.return_value = None
            
            main()
            
            captured = capsys.readouterr()
            assert "Skipping" in captured.out
            assert "could not calculate overall average KV cache usage" in captured.out
            assert "Skipped (no data or other reasons): 1 directories" in captured.out
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"])
    def test_main_file_write_error(self, mock_base_path, temp_dir, capsys):
        """Test main when file writing fails."""
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        # Create the directory structure
        model_dir = temp_dir / "TestModel"
        model_dir.mkdir()
        dataset_dir = model_dir / "test_dataset"
        dataset_dir.mkdir()
        sc_dir = dataset_dir / "sc_8_control"
        sc_dir.mkdir()
        
        # Mock calculate_overall_avg_kv_cache_for_run
        with patch('prune.utils.precompute.calculate_overall_avg_kv_cache_for_run') as mock_calc:
            mock_calc.return_value = 0.75
            
            # Mock file write to raise IOError
            with patch('builtins.open', side_effect=IOError("Permission denied")):
                main()
                
                captured = capsys.readouterr()
                assert "ERROR: Could not write to" in captured.out
                assert "Errors during writing: 1 directories" in captured.out
    
    @patch('prune.utils.precompute.BASE_RESULTS_PATH')
    @patch('prune.utils.precompute.MODELS_TO_PROCESS', ["Model1", "Model2"])
    @patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["dataset1", "dataset2"])
    def test_main_multiple_models_datasets(self, mock_base_path, temp_dir, capsys):
        """Test main with multiple models and datasets."""
        mock_base_path.__truediv__ = lambda self, other: temp_dir / other
        
        # Create multiple model/dataset combinations
        for model in ["Model1", "Model2"]:
            for dataset in ["dataset1", "dataset2"]:
                model_dir = temp_dir / model
                model_dir.mkdir(exist_ok=True)
                dataset_dir = model_dir / dataset
                dataset_dir.mkdir(exist_ok=True)
                
                # Create sc directories
                for sc_num in [4, 8]:
                    sc_dir = dataset_dir / f"sc_{sc_num}_control"
                    sc_dir.mkdir()
        
        # Mock calculate_overall_avg_kv_cache_for_run
        with patch('prune.utils.precompute.calculate_overall_avg_kv_cache_for_run') as mock_calc:
            mock_calc.return_value = 0.75
            
            # Mock file writing
            with patch('builtins.open', mock_open()):
                main()
                
                # Should process 2 models * 2 datasets * 2 sc_dirs = 8 directories
                captured = capsys.readouterr()
                assert "Successfully processed and saved: 8 directories" in captured.out
    
    @patch('prune.utils.precompute.OUTPUT_FILENAME', "test_output.txt")
    def test_main_output_filename_used(self, temp_dir, capsys):
        """Test that the correct output filename is used."""
        # This test verifies that the OUTPUT_FILENAME constant is used correctly
        # by checking that the patched value appears in the output
        
        with patch('prune.utils.precompute.BASE_RESULTS_PATH', temp_dir), \
             patch('prune.utils.precompute.MODELS_TO_PROCESS', ["TestModel"]), \
             patch('prune.utils.precompute.DATASETS_TO_PROCESS', ["test_dataset"]):
            
            # Create directory structure
            model_dir = temp_dir / "TestModel"
            model_dir.mkdir()
            dataset_dir = model_dir / "test_dataset"
            dataset_dir.mkdir()
            sc_dir = dataset_dir / "sc_8_control"
            sc_dir.mkdir()
            
            with patch('prune.utils.precompute.calculate_overall_avg_kv_cache_for_run') as mock_calc:
                mock_calc.return_value = 0.75
                
                with patch('builtins.open', mock_open()) as mock_file:
                    main()
                    
                    # Check that the output file was created with correct name
                    expected_path = sc_dir / "test_output.txt"
                    mock_file.assert_called_with(expected_path, 'w')
                    
                    captured = capsys.readouterr()
                    assert "test_output.txt" in captured.out


class TestPrecomputeIntegration:
    """Integration tests for the precompute module."""
    
    def test_full_pipeline(self, temp_dir, mock_json_file):
        """Test the full pipeline from JSON files to output file."""
        # Create realistic directory structure
        models_dir = temp_dir / "TestModel"
        models_dir.mkdir()
        dataset_dir = models_dir / "hmmt"
        dataset_dir.mkdir()
        sc_dir = dataset_dir / "sc_8_control"
        sc_dir.mkdir()
        summaries_dir = sc_dir / "summaries"
        summaries_dir.mkdir()
        
        # Create realistic summary JSON files
        test_data = [
            {"avg_kv_cache_usage": 0.75, "total_tokens": 1000, "question_id": 1},
            {"avg_kv_cache_usage": 0.85, "total_tokens": 1200, "question_id": 2},
            {"avg_kv_cache_usage": 0.65, "total_tokens": 800, "question_id": 3}
        ]
        
        for i, data in enumerate(test_data):
            file_path = mock_json_file(data, f"question_{i+1}_summary.json")
            new_path = summaries_dir / f"question_{i+1}_summary.json"
            file_path.rename(new_path)
        
        # Test the calculate function
        result = calculate_overall_avg_kv_cache_for_run(sc_dir)
        
        expected_mean = (0.75 + 0.85 + 0.65) / 3
        assert abs(result - expected_mean) < 1e-10
        
        # Test writing the output file
        output_file = sc_dir / "precomputed_mean_gpu_cache_perc.txt"
        with open(output_file, 'w') as f:
            f.write(f"{result:.10f}")
        
        # Verify the file was written correctly
        assert output_file.exists()
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert abs(float(content) - expected_mean) < 1e-10