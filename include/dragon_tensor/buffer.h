#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace dragon_tensor {

// Forward declarations
enum class DType;

// Buffer base class for memory management abstraction
class Buffer {
 public:
  virtual ~Buffer() = default;

  // Get pointer to underlying memory
  virtual void* data() = 0;
  virtual const void* data() const = 0;

  // Get size in bytes
  virtual size_t size_bytes() const = 0;

  // Memory management
  virtual void flush() {}
  virtual void detach() {}
};

// In-memory buffer (owns std::vector)
class MemoryBuffer : public Buffer {
 public:
  template <typename T>
  MemoryBuffer(const T* src, size_t count)
      : data_(new uint8_t[count * sizeof(T)]), size_bytes_(count * sizeof(T)) {
    std::memcpy(data_.get(), src, size_bytes_);
  }

  template <typename T>
  MemoryBuffer(size_t count)
      : data_(new uint8_t[count * sizeof(T)]), size_bytes_(count * sizeof(T)) {
    std::memset(data_.get(), 0, size_bytes_);
  }

  explicit MemoryBuffer(size_t size_bytes)
      : data_(new uint8_t[size_bytes]), size_bytes_(size_bytes) {
    std::memset(data_.get(), 0, size_bytes_);
  }

  void* data() override { return data_.get(); }
  const void* data() const override { return data_.get(); }
  size_t size_bytes() const override { return size_bytes_; }

 private:
  std::unique_ptr<uint8_t[]> data_;
  size_t size_bytes_;
};

// Memory-mapped file buffer
class MMapBuffer : public Buffer {
 public:
  MMapBuffer(const std::string& path, size_t offset, size_t size_bytes);
  ~MMapBuffer() override;

  // Disable copy
  MMapBuffer(const MMapBuffer&) = delete;
  MMapBuffer& operator=(const MMapBuffer&) = delete;

  // Enable move
  MMapBuffer(MMapBuffer&& other) noexcept;
  MMapBuffer& operator=(MMapBuffer&& other) noexcept;

  void* data() override { return mapped_ptr_; }
  const void* data() const override { return mapped_ptr_; }
  size_t size_bytes() const override { return size_bytes_; }

  void flush() override;
  void detach() override;

 private:
  void* mapped_ptr_;
  size_t size_bytes_;
  size_t mapped_size_;
  int fd_;
  bool is_valid_;
};

// Shared memory buffer (POSIX shared memory)
class SharedMemoryBuffer : public Buffer {
 public:
  static std::shared_ptr<SharedMemoryBuffer> create(const std::string& name,
                                                    size_t size_bytes);
  static std::shared_ptr<SharedMemoryBuffer> attach(const std::string& name);
  static void destroy(const std::string& name);

  ~SharedMemoryBuffer() override;

  // Disable copy
  SharedMemoryBuffer(const SharedMemoryBuffer&) = delete;
  SharedMemoryBuffer& operator=(const SharedMemoryBuffer&) = delete;

  void* data() override { return shared_ptr_; }
  const void* data() const override { return shared_ptr_; }
  size_t size_bytes() const override { return size_bytes_; }

  void detach() override;

  const std::string& name() const { return name_; }

 private:
  SharedMemoryBuffer(const std::string& name, void* ptr, size_t size_bytes,
                     int shm_fd);

  std::string name_;
  void* shared_ptr_;
  size_t size_bytes_;
  int shm_fd_;
  bool owns_;
};

}  // namespace dragon_tensor
