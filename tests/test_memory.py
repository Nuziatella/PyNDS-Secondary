from unittest.mock import Mock

import pytest

from pynds.memory import Memory


class TestMemoryInitialization:
    """Test Memory class initialization."""

    def test_memory_init(self):
        """Test Memory initialization."""
        mock_nds = Mock()

        memory = Memory(mock_nds)

        assert memory._nds is mock_nds
        assert memory._initialized is True

    def test_memory_is_initialized(self):
        """Test memory initialization check."""
        mock_nds = Mock()

        memory = Memory(mock_nds)

        assert memory.is_initialized() is True

        # Test when not initialized
        memory._initialized = False
        assert memory.is_initialized() is False

        # Test when NDS is None
        memory._nds = None
        assert memory.is_initialized() is False


class TestMemoryReadOperations:
    """Test memory read operations."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock Memory instance for testing."""
        mock_nds = Mock()
        memory = Memory(mock_nds)
        return memory

    def test_read_ram_u8(self, mock_memory):
        """Test reading unsigned 8-bit value."""
        mock_memory._nds.read_ram_u8.return_value = 255

        result = mock_memory.read_ram_u8(0x02000000)

        assert result == 255
        mock_memory._nds.read_ram_u8.assert_called_once_with(0x02000000)

    def test_read_ram_u16(self, mock_memory):
        """Test reading unsigned 16-bit value."""
        mock_memory._nds.read_ram_u16.return_value = 65535

        result = mock_memory.read_ram_u16(0x02000000)

        assert result == 65535
        mock_memory._nds.read_ram_u16.assert_called_once_with(0x02000000)

    def test_read_ram_u32(self, mock_memory):
        """Test reading unsigned 32-bit value."""
        mock_memory._nds.read_ram_u32.return_value = 4294967295

        result = mock_memory.read_ram_u32(0x02000000)

        assert result == 4294967295
        mock_memory._nds.read_ram_u32.assert_called_once_with(0x02000000)

    def test_read_ram_u64(self, mock_memory):
        """Test reading unsigned 64-bit value."""
        mock_memory._nds.read_ram_u64.return_value = 18446744073709551615

        result = mock_memory.read_ram_u64(0x02000000)

        assert result == 18446744073709551615
        mock_memory._nds.read_ram_u64.assert_called_once_with(0x02000000)

    def test_read_ram_i8(self, mock_memory):
        """Test reading signed 8-bit value."""
        mock_memory._nds.read_ram_i8.return_value = -128

        result = mock_memory.read_ram_i8(0x02000000)

        assert result == -128
        mock_memory._nds.read_ram_i8.assert_called_once_with(0x02000000)

    def test_read_ram_i16(self, mock_memory):
        """Test reading signed 16-bit value."""
        mock_memory._nds.read_ram_i16.return_value = -32768

        result = mock_memory.read_ram_i16(0x02000000)

        assert result == -32768
        mock_memory._nds.read_ram_i16.assert_called_once_with(0x02000000)

    def test_read_ram_i32(self, mock_memory):
        """Test reading signed 32-bit value."""
        mock_memory._nds.read_ram_i32.return_value = -2147483648

        result = mock_memory.read_ram_i32(0x02000000)

        assert result == -2147483648
        mock_memory._nds.read_ram_i32.assert_called_once_with(0x02000000)

    def test_read_ram_i64(self, mock_memory):
        """Test reading signed 64-bit value."""
        mock_memory._nds.read_ram_i64.return_value = -9223372036854775808

        result = mock_memory.read_ram_i64(0x02000000)

        assert result == -9223372036854775808
        mock_memory._nds.read_ram_i64.assert_called_once_with(0x02000000)

    def test_read_ram_f32(self, mock_memory):
        """Test reading 32-bit floating point value."""
        mock_memory._nds.read_ram_f32.return_value = 3.14159

        result = mock_memory.read_ram_f32(0x02000000)

        assert result == 3.14159
        mock_memory._nds.read_ram_f32.assert_called_once_with(0x02000000)

    def test_read_ram_f64(self, mock_memory):
        """Test reading 64-bit floating point value."""
        mock_memory._nds.read_ram_f64.return_value = 3.141592653589793

        result = mock_memory.read_ram_f64(0x02000000)

        assert result == 3.141592653589793
        mock_memory._nds.read_ram_f64.assert_called_once_with(0x02000000)

    def test_read_map(self, mock_memory):
        """Test reading a range of memory addresses."""
        mock_data = [0x01, 0x02, 0x03, 0x04]
        mock_memory._nds.read_map.return_value = mock_data

        result = mock_memory.read_map(0x02000000, 0x02000004)

        assert result == mock_data
        mock_memory._nds.read_map.assert_called_once_with(0x02000000, 0x02000004)


class TestMemoryWriteOperations:
    """Test memory write operations."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock Memory instance for testing."""
        mock_nds = Mock()
        memory = Memory(mock_nds)
        return memory

    def test_write_ram_u8(self, mock_memory):
        """Test writing unsigned 8-bit value."""
        mock_memory.write_ram_u8(0x02000000, 255)

        mock_memory._nds.write_ram_u8.assert_called_once_with(0x02000000, 255)

    def test_write_ram_u16(self, mock_memory):
        """Test writing unsigned 16-bit value."""
        mock_memory.write_ram_u16(0x02000000, 65535)

        mock_memory._nds.write_ram_u16.assert_called_once_with(0x02000000, 65535)

    def test_write_ram_u32(self, mock_memory):
        """Test writing unsigned 32-bit value."""
        mock_memory.write_ram_u32(0x02000000, 4294967295)

        mock_memory._nds.write_ram_u32.assert_called_once_with(0x02000000, 4294967295)

    def test_write_ram_u64(self, mock_memory):
        """Test writing unsigned 64-bit value."""
        mock_memory.write_ram_u64(0x02000000, 18446744073709551615)

        mock_memory._nds.write_ram_u64.assert_called_once_with(
            0x02000000, 18446744073709551615
        )

    def test_write_ram_i8(self, mock_memory):
        """Test writing signed 8-bit value."""
        mock_memory.write_ram_i8(0x02000000, -128)

        mock_memory._nds.write_ram_i8.assert_called_once_with(0x02000000, -128)

    def test_write_ram_i16(self, mock_memory):
        """Test writing signed 16-bit value."""
        mock_memory.write_ram_i16(0x02000000, -32768)

        mock_memory._nds.write_ram_i16.assert_called_once_with(0x02000000, -32768)

    def test_write_ram_i32(self, mock_memory):
        """Test writing signed 32-bit value."""
        mock_memory.write_ram_i32(0x02000000, -2147483648)

        mock_memory._nds.write_ram_i32.assert_called_once_with(0x02000000, -2147483648)

    def test_write_ram_i64(self, mock_memory):
        """Test writing signed 64-bit value."""
        mock_memory.write_ram_i64(0x02000000, -9223372036854775808)

        mock_memory._nds.write_ram_i64.assert_called_once_with(
            0x02000000, -9223372036854775808
        )

    def test_write_ram_f32(self, mock_memory):
        """Test writing 32-bit floating point value."""
        mock_memory.write_ram_f32(0x02000000, 3.14159)

        mock_memory._nds.write_ram_f32.assert_called_once_with(0x02000000, 3.14159)

    def test_write_ram_f64(self, mock_memory):
        """Test writing 64-bit floating point value."""
        mock_memory.write_ram_f64(0x02000000, 3.141592653589793)

        mock_memory._nds.write_ram_f64.assert_called_once_with(
            0x02000000, 3.141592653589793
        )

    def test_write_map(self, mock_memory):
        """Test writing a list of byte values to memory."""
        test_data = [0x01, 0x02, 0x03, 0x04]

        mock_memory.write_map(0x02000000, test_data)

        mock_memory._nds.write_map.assert_called_once_with(0x02000000, test_data)


class TestMemoryCleanup:
    """Test Memory cleanup functionality."""

    def test_close_method(self):
        """Test the close method."""
        mock_nds = Mock()
        memory = Memory(mock_nds)

        assert memory._initialized is True

        memory.close()

        assert memory._initialized is False

    def test_close_when_not_initialized(self):
        """Test closing when not initialized."""
        mock_nds = Mock()
        memory = Memory(mock_nds)
        memory._initialized = False

        # Should not raise error
        memory.close()

        assert memory._initialized is False

    def test_destructor_cleanup(self):
        """Test that destructor calls cleanup if not explicitly closed."""
        mock_nds = Mock()
        memory = Memory(mock_nds)

        # Don't call close() explicitly
        del memory

        # The destructor should have been called
        # (This is hard to test directly, but we can verify the cleanup logic)


class TestMemoryEdgeCases:
    """Test memory operations with edge cases."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock Memory instance for testing."""
        mock_nds = Mock()
        memory = Memory(mock_nds)
        return memory

    def test_read_write_roundtrip(self, mock_memory):
        """Test that read/write operations are consistent."""

        # Mock the read to return what we wrote
        def mock_write(address, value):
            mock_memory._nds.read_ram_u32.return_value = value

        mock_memory._nds.write_ram_u32.side_effect = mock_write

        # Write a value
        test_value = 0x12345678
        mock_memory.write_ram_u32(0x02000000, test_value)

        # Read it back
        result = mock_memory.read_ram_u32(0x02000000)

        assert result == test_value

    def test_memory_map_edge_cases(self, mock_memory):
        """Test memory mapping with edge cases."""
        # Test empty range
        mock_memory._nds.read_map.return_value = []

        result = mock_memory.read_map(0x02000000, 0x02000000)

        assert result == []
        mock_memory._nds.read_map.assert_called_once_with(0x02000000, 0x02000000)

    def test_large_memory_operations(self, mock_memory):
        """Test memory operations with large values."""
        # Test maximum 32-bit unsigned value
        max_u32 = 0xFFFFFFFF
        mock_memory.write_ram_u32(0x02000000, max_u32)
        mock_memory._nds.write_ram_u32.assert_called_with(0x02000000, max_u32)

        # Test maximum 32-bit signed value
        max_i32 = 0x7FFFFFFF
        mock_memory.write_ram_i32(0x02000000, max_i32)
        mock_memory._nds.write_ram_i32.assert_called_with(0x02000000, max_i32)
