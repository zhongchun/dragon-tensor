#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "dragon_tensor/backend.h"

namespace dragon_tensor {

// Forward declarations
class Buffer;
enum class Layout;

// Arrow/Parquet backend for columnar analytics
// This backend provides storage via Apache Arrow format and Parquet files
class ArrowBackend : public Backend {
 public:
  explicit ArrowBackend(std::string_view path = "", bool read_only = false);
  ~ArrowBackend() override;

  [[nodiscard]] std::shared_ptr<Buffer> allocate(size_t size_bytes,
                                                 Layout layout) override;

  void release(Buffer& buffer) override;

  [[nodiscard]] std::string name() const override { return "ArrowBackend"; }

  void flush() override;

  [[nodiscard]] bool supports_mmap() const override { return true; }

  [[nodiscard]] bool is_writable() const override { return !read_only_; }
  [[nodiscard]] bool is_readable() const override { return true; }

  // Arrow-specific methods
  // std::shared_ptr<arrow::RecordBatch> read_batch(size_t offset, size_t
  // length); void write_batch(const arrow::RecordBatch& batch);

 private:
  std::string path_;
  bool read_only_;
  // Arrow-specific data members will be added when Arrow library is integrated
  // std::shared_ptr<arrow::MemoryPool> pool_;
  // std::shared_ptr<arrow::io::RandomAccessFile> file_;
};

// Factory function for Arrow backend (declared in backend.h)

}  // namespace dragon_tensor
