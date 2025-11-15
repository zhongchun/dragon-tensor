"""
Tests for I/O operations
"""

import os
import numpy as np
import pytest
import dragon_tensor as dt


class TestSaveLoad:
    """Test save and load operations"""

    def test_save_load_1d(self, sample_1d_tensor, temp_dir):
        """Test saving and loading 1D tensor"""
        file_path = temp_dir / "test_1d.bin"
        dt.save(sample_1d_tensor, str(file_path))
        assert file_path.exists()

        # Load using TensorDouble.load() since we saved a double tensor
        loaded = dt.TensorDouble.load(str(file_path), mmap=False)
        original = dt.to_numpy(sample_1d_tensor)
        result = dt.to_numpy(loaded)
        np.testing.assert_array_almost_equal(result, original)

    def test_save_load_2d(self, sample_2d_tensor, temp_dir):
        """Test saving and loading 2D tensor"""
        file_path = temp_dir / "test_2d.bin"
        dt.save(sample_2d_tensor, str(file_path))
        assert file_path.exists()

        # Load using TensorDouble.load() since we saved a double tensor
        loaded = dt.TensorDouble.load(str(file_path), mmap=False)
        original = dt.to_numpy(sample_2d_tensor)
        result = dt.to_numpy(loaded)
        np.testing.assert_array_almost_equal(result, original)

    def test_save_load_with_layout(self, sample_2d_tensor, temp_dir):
        """Test saving with different layouts"""
        file_path_row = temp_dir / "test_row.bin"
        file_path_col = temp_dir / "test_col.bin"

        dt.save(sample_2d_tensor, str(file_path_row), layout="row")
        dt.save(sample_2d_tensor, str(file_path_col), layout="column")

        assert file_path_row.exists()
        assert file_path_col.exists()

        # Both should load correctly using TensorDouble.load()
        loaded_row = dt.TensorDouble.load(str(file_path_row), mmap=False)
        loaded_col = dt.TensorDouble.load(str(file_path_col), mmap=False)

        original = dt.to_numpy(sample_2d_tensor)
        result_row = dt.to_numpy(loaded_row)
        result_col = dt.to_numpy(loaded_col)

        np.testing.assert_array_almost_equal(result_row, original)
        np.testing.assert_array_almost_equal(result_col, original)

    def test_open_mmap(self, sample_1d_tensor, temp_dir):
        """Test opening file with memory mapping"""
        file_path = temp_dir / "test_mmap.bin"
        dt.save(sample_1d_tensor, str(file_path))

        # dt.open() is a context manager
        with dt.open(str(file_path), mmap=True) as opened:
            original = dt.to_numpy(sample_1d_tensor)
            result = dt.to_numpy(opened)
            np.testing.assert_array_almost_equal(result, original)

    def test_open_no_mmap(self, sample_1d_tensor, temp_dir):
        """Test opening file without memory mapping"""
        file_path = temp_dir / "test_no_mmap.bin"
        dt.save(sample_1d_tensor, str(file_path))

        # dt.open() is a context manager
        with dt.open(str(file_path), mmap=False) as opened:
            original = dt.to_numpy(sample_1d_tensor)
            result = dt.to_numpy(opened)
            np.testing.assert_array_almost_equal(result, original)

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises error"""
        file_path = temp_dir / "nonexistent.bin"
        with pytest.raises((FileNotFoundError, IOError, RuntimeError)):
            dt.TensorDouble.load(str(file_path), mmap=False)

    @pytest.mark.requires_arrow
    def test_save_parquet_not_implemented(self, sample_1d_tensor, temp_dir):
        """Test that save_parquet raises NotImplementedError"""
        file_path = temp_dir / "test.parquet"
        with pytest.raises(NotImplementedError):
            dt.save_parquet(sample_1d_tensor, str(file_path))

    @pytest.mark.requires_arrow
    def test_load_parquet_not_implemented(self, temp_dir):
        """Test that load_parquet raises NotImplementedError"""
        file_path = temp_dir / "test.parquet"
        with pytest.raises(NotImplementedError):
            dt.load_parquet(str(file_path))
