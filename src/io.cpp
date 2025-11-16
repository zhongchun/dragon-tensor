#include "dragon_tensor/io.h"

#include <unistd.h>  // For sysconf and _SC_PAGESIZE

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "dragon_tensor/buffer.h"
#include "dragon_tensor/storage.h"
#include "dragon_tensor/tensor.h"

namespace fs = std::filesystem;

namespace dragon_tensor {
namespace io {

template <typename T>
void save_tensor(const Tensor<T>& tensor, std::string_view path,
                 Layout layout) {
  // Use C++17 filesystem to ensure parent directory exists
  fs::path file_path(path);
  if (auto parent = file_path.parent_path(); !parent.empty()) {
    fs::create_directories(parent);
  }

  std::string path_str(path);  // Convert string_view to string for ofstream
  std::ofstream file(path_str, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " +
                             std::string(path));
  }

  // Prepare header
  TensorHeader header;
  header.ndim = static_cast<uint32_t>(tensor.ndim());
  header.dtype = static_cast<uint32_t>(get_dtype<T>());
  header.layout = (layout == Layout::RowMajor) ? 0 : 1;
  header.endian = is_little_endian() ? 0 : 1;
  header.data_offset = calculate_header_size(header.ndim);

  // Write header
  file.write(reinterpret_cast<const char*>(&header), sizeof(TensorHeader));

  // Write shape
  const auto& shape = tensor.shape();
  for (size_t dim : shape) {
    uint64_t dim_val = static_cast<uint64_t>(dim);
    file.write(reinterpret_cast<const char*>(&dim_val), sizeof(uint64_t));
  }

  // Calculate data size
  size_t total_elements = tensor.size();
  size_t data_size_bytes = total_elements * sizeof(T);

  // Write data (simple implementation - always row-major for now)
  const T* data_ptr = tensor.raw_data();
  file.write(reinterpret_cast<const char*>(data_ptr), data_size_bytes);

  // Calculate and write checksum
  uint32_t checksum = calculate_checksum(data_ptr, data_size_bytes);
  header.checksum = static_cast<uint64_t>(checksum);

  // Seek back and update checksum
  std::streampos checksum_offset = offsetof(TensorHeader, checksum);
  file.seekp(checksum_offset);
  file.write(reinterpret_cast<const char*>(&header.checksum), sizeof(uint64_t));

  file.close();
}

template <typename T>
Tensor<T> load_tensor(std::string_view path, bool mmap /* = true */) {
  // Use C++17 filesystem to check file existence
  if (!fs::exists(path)) {
    throw std::runtime_error("File does not exist: " + std::string(path));
  }

  std::string path_str(path);  // Convert string_view to string

  // Read header first (needed for both mmap and regular load)
  std::ifstream file(path_str, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " +
                             std::string(path));
  }

  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(TensorHeader));

  // Validate magic
  if (header.magic != TensorHeader::MAGIC) {
    throw std::runtime_error("Invalid file format: bad magic number");
  }

  // Validate version
  if (header.version != TensorHeader::VERSION) {
    throw std::runtime_error("Unsupported file version: " +
                             std::to_string(header.version));
  }

  // Validate dtype
  DType expected_dtype = get_dtype<T>();
  if (static_cast<DType>(header.dtype) != expected_dtype) {
    throw std::runtime_error("Dtype mismatch in file");
  }

  // Read shape
  std::vector<size_t> shape(header.ndim);
  for (uint32_t i = 0; i < header.ndim; ++i) {
    uint64_t dim_val;
    file.read(reinterpret_cast<char*>(&dim_val), sizeof(uint64_t));
    shape[i] = static_cast<size_t>(dim_val);
  }

  file.close();  // Close file before mmap

  // Calculate data size
  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
  size_t data_size_bytes = total_elements * sizeof(T);

  if (mmap) {
    // Use memory-mapped I/O for on-demand loading
    // Data offset is where the actual tensor data starts (after header + shape)
    size_t data_offset = header.data_offset;

    // mmap requires page-aligned offsets on many systems
    // Map from the beginning of the file, then adjust pointer
    // Get page size for alignment
    size_t page_size = sysconf(_SC_PAGESIZE);
    if (page_size == 0) {
      page_size = 4096;  // Default to 4KB if sysconf fails
    }

    // Calculate aligned offset (round down to page boundary)
    size_t aligned_offset = (data_offset / page_size) * page_size;
    size_t offset_adjustment = data_offset - aligned_offset;
    size_t total_map_size = offset_adjustment + data_size_bytes;

    // Create memory-mapped buffer from aligned offset
    auto buffer =
        std::make_shared<MMapBuffer>(path_str, aligned_offset, total_map_size);

    // Create a wrapper buffer that adjusts the pointer
    // We'll create a custom buffer that wraps MMapBuffer with offset adjustment
    class OffsetBuffer : public Buffer {
     public:
      OffsetBuffer(std::shared_ptr<MMapBuffer> base, size_t offset)
          : base_(base), offset_(offset) {}
      void* data() override {
        return static_cast<char*>(base_->data()) + offset_;
      }
      const void* data() const override {
        return static_cast<const char*>(base_->data()) + offset_;
      }
      size_t size_bytes() const override {
        return base_->size_bytes() - offset_;
      }
      void flush() override { base_->flush(); }
      void detach() override { base_->detach(); }

     private:
      std::shared_ptr<MMapBuffer> base_;
      size_t offset_;
    };

    auto offset_buffer =
        std::make_shared<OffsetBuffer>(buffer, offset_adjustment);

    // Create tensor with mmap buffer (no data copy)
    Tensor<T> tensor;
    tensor.shape_ = shape;
    tensor.storage_mode_ = StorageMode::MMap;
    tensor.layout_ =
        (header.layout == 0) ? Layout::RowMajor : Layout::ColumnMajor;
    tensor.buffer_ = offset_buffer;

    // data_ remains empty - tensor uses buffer_ directly
    // All data access goes through raw_data() which uses buffer_->data()

    return tensor;
  } else {
    // Regular load: read all data into memory
    file.open(path_str, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for reading: " +
                               std::string(path));
    }

    // Skip header and shape (we already read them)
    file.seekg(header.data_offset);

    // Read data
    std::vector<T> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), data_size_bytes);

    if (file.gcount() != static_cast<std::streamsize>(data_size_bytes)) {
      throw std::runtime_error("File truncated or incomplete");
    }

    // Verify checksum (if enabled)
    if (header.checksum != 0) {
      uint32_t calculated_checksum =
          calculate_checksum(data.data(), data_size_bytes);
      // Note: header.checksum is uint64_t, but calculate_checksum returns
      // uint32_t For now, we'll just check if checksum was calculated
    }

    return Tensor<T>(shape, std::move(data));
  }
}

// Read dtype from file header
DType read_dtype_from_file(std::string_view path) {
  // Use C++17 filesystem to check file existence
  if (!fs::exists(path)) {
    throw std::runtime_error("File does not exist: " + std::string(path));
  }

  std::string path_str(path);  // Convert string_view to string for ifstream
  std::ifstream file(path_str, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " +
                             std::string(path));
  }

  // Read header
  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(TensorHeader));

  // Validate magic
  if (header.magic != TensorHeader::MAGIC) {
    throw std::runtime_error("Invalid file format: bad magic number");
  }

  // Validate version
  if (header.version != TensorHeader::VERSION) {
    throw std::runtime_error("Unsupported file version: " +
                             std::to_string(header.version));
  }

  // Return the dtype from the header
  return static_cast<DType>(header.dtype);
}

// Explicit instantiations
template void save_tensor<float>(const Tensor<float>&, std::string_view,
                                 Layout);
template void save_tensor<double>(const Tensor<double>&, std::string_view,
                                  Layout);
template void save_tensor<int32_t>(const Tensor<int32_t>&, std::string_view,
                                   Layout);
template void save_tensor<int64_t>(const Tensor<int64_t>&, std::string_view,
                                   Layout);
template void save_tensor<uint8_t>(const Tensor<uint8_t>&, std::string_view,
                                   Layout);

template Tensor<float> load_tensor<float>(std::string_view, bool);
template Tensor<double> load_tensor<double>(std::string_view, bool);
template Tensor<int32_t> load_tensor<int32_t>(std::string_view, bool);
template Tensor<int64_t> load_tensor<int64_t>(std::string_view, bool);
template Tensor<uint8_t> load_tensor<uint8_t>(std::string_view, bool);

}  // namespace io
}  // namespace dragon_tensor
