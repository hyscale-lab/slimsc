"""Tests for prune.utils.unzip_results module."""

import pytest
from unittest.mock import patch, Mock, MagicMock, mock_open
import zipfile
from pathlib import Path
import sys

from prune.utils.unzip_results import unzip_results


class TestUnzipResults:
    """Test the unzip_results function."""
    
    @patch('prune.utils.unzip_results.zipfile.ZipFile')
    @patch('prune.utils.unzip_results.Path.exists')
    @patch('prune.utils.unzip_results.sys.exit')
    def test_unzip_results_success(self, mock_exit, mock_exists, mock_zipfile, temp_dir, capsys):
        """Test successful unzipping of results."""
        # Setup actual paths using temp_dir
        script_dir = temp_dir / "utils"
        script_dir.mkdir()
        base_results_dir = temp_dir / "results"
        base_results_dir.mkdir()
        target_parent_dir = base_results_dir / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        zip_file_path = target_parent_dir / "test_dir.zip"
        
        # Create a real zip file for testing
        with zipfile.ZipFile(zip_file_path, 'w') as zf:
            zf.writestr("test_dir/file1.txt", "content1")
            zf.writestr("test_dir/file2.txt", "content2")
            zf.writestr("test_dir/subdir/file3.txt", "content3")
        
        # Mock Path resolution to use our temp directory structure
        with patch('prune.utils.unzip_results.Path') as mock_path_class:
            def path_constructor(path_str):
                if path_str == "__file__":
                    return script_dir / "unzip_results.py"
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_constructor
            
            # Mock zipfile operations to count extractions
            mock_zip_context = Mock()
            mock_zip_context.__enter__ = Mock(return_value=mock_zip_context)
            mock_zip_context.__exit__ = Mock(return_value=False)
            mock_zip_context.namelist.return_value = [
                "test_dir/file1.txt", "test_dir/file2.txt", "test_dir/subdir/", "test_dir/subdir/file3.txt"
            ]
            mock_zipfile.return_value = mock_zip_context
            
            # Mock file existence checks (all files are new)
            mock_exists.return_value = False
            
            unzip_results("TestModel", "test_dataset", "test_dir", verbose=True)
            
            # Verify zipfile was opened and extract was called for files (not directories)
            mock_zipfile.assert_called_once()
            assert mock_zip_context.extract.call_count == 3  # 3 files (not the directory entry)
            
            # Check console output
            captured = capsys.readouterr()
            assert "--- Unzipping Results ---" in captured.out
            assert "Model: TestModel, Dataset: test_dataset, Directory: test_dir" in captured.out
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_directory_not_found(self, mock_path, temp_dir, capsys):
        """Test when parent directory doesn't exist."""
        script_dir = temp_dir / "utils"
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        # Mock directory as not existing
        type(mock_path.return_value).is_dir = property(lambda self: False)
        
        with pytest.raises(SystemExit) as exc_info:
            unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Parent directory not found" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_zip_file_not_found(self, mock_path, temp_dir, capsys):
        """Test when ZIP file doesn't exist."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        # Mock directory exists but zip file doesn't
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return False  # ZIP file doesn't exist
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        with pytest.raises(SystemExit) as exc_info:
            unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: ZIP file not found" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_skip_existing_files(self, mock_path, temp_dir, capsys):
        """Test skipping existing files during extraction."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        zip_file_path = target_parent_dir / "test_dir.zip"
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        mock_zip_members = ["test_dir/file1.txt", "test_dir/file2.txt", "test_dir/subdir/"]
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_ref = MagicMock()
            mock_zip_ref.namelist.return_value = mock_zip_members
            mock_zip_ref.__enter__ = Mock(return_value=mock_zip_ref)
            mock_zip_ref.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_ref
            
            # Mock that file1.txt already exists, but file2.txt doesn't
            def mock_exists(self):
                return "file1.txt" in str(self)
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=True)
            
            # Should only extract file2.txt (file1.txt is skipped, subdir/ is a directory)
            assert mock_zip_ref.extract.call_count == 1
            
            captured = capsys.readouterr()
            assert "Skipping existing file" in captured.out
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_empty_zip(self, mock_path, temp_dir, capsys):
        """Test handling of empty ZIP file."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_ref = MagicMock()
            mock_zip_ref.namelist.return_value = []  # Empty ZIP
            mock_zip_ref.__enter__ = Mock(return_value=mock_zip_ref)
            mock_zip_ref.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_ref
            
            unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            captured = capsys.readouterr()
            assert "Warning: ZIP file" in captured.err
            assert "is empty" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_invalid_zip_file(self, mock_path, temp_dir, capsys):
        """Test handling of corrupted ZIP file."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        # Mock zipfile to raise BadZipFile
        with patch('zipfile.ZipFile', side_effect=zipfile.BadZipFile("Corrupted ZIP")):
            with pytest.raises(SystemExit) as exc_info:
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error:" in captured.err
            assert "is not a valid ZIP file or is corrupted" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_file_not_found_during_extraction(self, mock_path, temp_dir, capsys):
        """Test handling when ZIP file is deleted during processing."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        # Mock zipfile to raise FileNotFoundError
        with patch('zipfile.ZipFile', side_effect=FileNotFoundError("ZIP file not found")):
            with pytest.raises(SystemExit) as exc_info:
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: ZIP file not found or inaccessible" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_unexpected_exception(self, mock_path, temp_dir, capsys):
        """Test handling of unexpected exceptions."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        # Mock zipfile to raise unexpected exception
        with patch('zipfile.ZipFile', side_effect=Exception("Unexpected error")):
            with pytest.raises(SystemExit) as exc_info:
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "An unexpected error occurred during unzipping" in captured.err
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_verbose_output(self, mock_path, temp_dir, capsys):
        """Test verbose output mode."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        mock_zip_members = ["test_dir/file1.txt", "test_dir/subdir/"]
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_ref = MagicMock()
            mock_zip_ref.namelist.return_value = mock_zip_members
            mock_zip_ref.__enter__ = Mock(return_value=mock_zip_ref)
            mock_zip_ref.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_ref
            
            with patch('pathlib.Path.exists', return_value=False):
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=True)
            
            captured = capsys.readouterr()
            assert "Base Results Dir:" in captured.out
            assert "Target Parent Dir:" in captured.out
            assert "ZIP File Path:" in captured.out
            assert "Extraction Root:" in captured.out
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_non_verbose_output(self, mock_path, temp_dir, capsys):
        """Test non-verbose output mode."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        mock_zip_members = ["test_dir/file1.txt"]
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_ref = MagicMock()
            mock_zip_ref.namelist.return_value = mock_zip_members
            mock_zip_ref.__enter__ = Mock(return_value=mock_zip_ref)
            mock_zip_ref.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_ref
            
            with patch('pathlib.Path.exists', return_value=False):
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            captured = capsys.readouterr()
            # Should not have verbose details
            assert "Base Results Dir:" not in captured.out
            assert "Target Parent Dir:" not in captured.out
            # But should have basic output
            assert "--- Unzipping Results ---" in captured.out
    
    @patch('prune.utils.unzip_results.Path')
    def test_unzip_results_extraction_count_reporting(self, mock_path, temp_dir, capsys):
        """Test that extraction and skip counts are reported correctly."""
        script_dir = temp_dir / "utils"
        target_parent_dir = temp_dir / "results" / "TestModel" / "test_dataset"
        target_parent_dir.mkdir(parents=True)
        
        mock_path.return_value.resolve.return_value.parent = script_dir
        
        def mock_is_dir(self):
            return "TestModel/test_dataset" in str(self)
        
        def mock_is_file(self):
            return "test_dir.zip" in str(self)
            
        type(mock_path.return_value).is_dir = property(lambda self: mock_is_dir(self))
        type(mock_path.return_value).is_file = property(lambda self: mock_is_file(self))
        
        # Mix of files and directories, some existing
        mock_zip_members = [
            "test_dir/file1.txt",    # Will be extracted
            "test_dir/file2.txt",    # Will be skipped (exists)
            "test_dir/subdir/",      # Directory (not counted as file)
            "test_dir/file3.txt"     # Will be extracted
        ]
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_ref = MagicMock()
            mock_zip_ref.namelist.return_value = mock_zip_members
            mock_zip_ref.__enter__ = Mock(return_value=mock_zip_ref)
            mock_zip_ref.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_ref
            
            # Mock that file2.txt already exists
            def mock_exists(self):
                return "file2.txt" in str(self)
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                unzip_results("TestModel", "test_dataset", "test_dir", verbose=False)
            
            # Should extract file1.txt and file3.txt (2 files)
            # Should skip file2.txt (1 file)
            # Should not count subdir/ (it's a directory)
            assert mock_zip_ref.extract.call_count == 2
            
            captured = capsys.readouterr()
            assert "Extraction completed successfully" in captured.out
            assert "Extracted: 2 files" in captured.out
            assert "Skipped: 1 files" in captured.out


class TestUnzipResultsArgumentHandling:
    """Test argument handling and validation."""
    
    def test_unzip_results_with_all_parameters(self, temp_dir):
        """Test that all parameters are handled correctly."""
        # This is more of a smoke test to ensure the function signature works
        with patch('prune.utils.unzip_results.Path') as mock_path:
            mock_path.return_value.resolve.return_value.parent = temp_dir
            type(mock_path.return_value).is_dir = property(lambda self: False)
            
            with pytest.raises(SystemExit):
                unzip_results(
                    model="TestModel", 
                    dataset="test_dataset", 
                    dir_name="test_dir", 
                    verbose=True
                )
    
    def test_unzip_results_parameter_types(self, temp_dir):
        """Test that parameter types are handled correctly."""
        with patch('prune.utils.unzip_results.Path') as mock_path:
            mock_path.return_value.resolve.return_value.parent = temp_dir
            type(mock_path.return_value).is_dir = property(lambda self: False)
            
            # Test with different parameter types
            with pytest.raises(SystemExit):
                unzip_results("TestModel", "test_dataset", "test_dir", False)
            
            with pytest.raises(SystemExit):
                unzip_results("TestModel", "test_dataset", "test_dir", True)


class TestUnzipResultsIntegration:
    """Integration tests for unzip_results function."""
    
    def test_full_unzip_pipeline(self, temp_dir):
        """Test the complete unzipping pipeline with real files."""
        # Create a realistic directory structure
        utils_dir = temp_dir / "utils"
        utils_dir.mkdir()
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        model_dir = results_dir / "TestModel"
        model_dir.mkdir()
        dataset_dir = model_dir / "test_dataset"
        dataset_dir.mkdir()
        
        # Create a real ZIP file with test content
        zip_path = dataset_dir / "test_run.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_run/config.json", '{"setting": "value"}')
            zf.writestr("test_run/results.txt", "Test results content")
            zf.writestr("test_run/logs/", "")  # Directory entry
            zf.writestr("test_run/logs/output.log", "Log content")
        
        # Mock the Path resolution to point to our test structure
        with patch('prune.utils.unzip_results.Path') as mock_path:
            # Mock __file__ resolution
            def path_side_effect(path_str):
                if path_str == "__file__":
                    mock_file_path = Mock()
                    mock_file_path.resolve.return_value.parent = utils_dir
                    return mock_file_path
                else:
                    # For other path operations, return actual Path objects
                    return Path(path_str)
            
            mock_path.side_effect = path_side_effect
            
            # Mock the base results directory resolution
            with patch.object(Path, '__truediv__') as mock_truediv:
                def truediv_side_effect(self, other):
                    if str(self).endswith("utils") and other == "../results":
                        return results_dir
                    else:
                        return Path(str(self)) / other
                
                mock_truediv.side_effect = truediv_side_effect
                
                # Run the unzip operation
                unzip_results("TestModel", "test_dataset", "test_run", verbose=False)
                
                # Verify that files were extracted
                assert (dataset_dir / "test_run" / "config.json").exists()
                assert (dataset_dir / "test_run" / "results.txt").exists()
                assert (dataset_dir / "test_run" / "logs" / "output.log").exists()
                
                # Verify file contents
                with open(dataset_dir / "test_run" / "config.json") as f:
                    assert '"setting": "value"' in f.read()
                
                with open(dataset_dir / "test_run" / "results.txt") as f:
                    assert "Test results content" == f.read()