#include "dragon_tensor/buffer.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace dragon_tensor {

// MMapBuffer implementation
MMapBuffer::MMapBuffer(const std::string& path, size_t offset,
                       size_t size_bytes)
    : mapped_ptr_(nullptr),
      size_bytes_(size_bytes),
      mapped_size_(0),
      fd_(-1),
      is_valid_(false) {
  // Open file
  fd_ = open(path.c_str(), O_RDWR);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file for mmap: " + path);
  }

  // Get file size
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    close(fd_);
    throw std::runtime_error("Failed to stat file: " + path);
  }

  size_t file_size = static_cast<size_t>(st.st_size);
  if (offset + size_bytes > file_size) {
    close(fd_);
    throw std::runtime_error("MMap range exceeds file size");
  }

  // Memory map
  mapped_ptr_ = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd_, offset);
  if (mapped_ptr_ == MAP_FAILED) {
    close(fd_);
    throw std::runtime_error("Failed to mmap file: " + path);
  }

  mapped_size_ = size_bytes;
  is_valid_ = true;
}

MMapBuffer::~MMapBuffer() { detach(); }

MMapBuffer::MMapBuffer(MMapBuffer&& other) noexcept
    : mapped_ptr_(other.mapped_ptr_),
      size_bytes_(other.size_bytes_),
      mapped_size_(other.mapped_size_),
      fd_(other.fd_),
      is_valid_(other.is_valid_) {
  other.mapped_ptr_ = nullptr;
  other.size_bytes_ = 0;
  other.mapped_size_ = 0;
  other.fd_ = -1;
  other.is_valid_ = false;
}

MMapBuffer& MMapBuffer::operator=(MMapBuffer&& other) noexcept {
  if (this != &other) {
    detach();
    mapped_ptr_ = other.mapped_ptr_;
    size_bytes_ = other.size_bytes_;
    mapped_size_ = other.mapped_size_;
    fd_ = other.fd_;
    is_valid_ = other.is_valid_;

    other.mapped_ptr_ = nullptr;
    other.size_bytes_ = 0;
    other.mapped_size_ = 0;
    other.fd_ = -1;
    other.is_valid_ = false;
  }
  return *this;
}

void MMapBuffer::flush() {
  if (is_valid_ && mapped_ptr_) {
    msync(mapped_ptr_, mapped_size_, MS_SYNC);
  }
}

void MMapBuffer::detach() {
  if (is_valid_ && mapped_ptr_) {
    munmap(mapped_ptr_, mapped_size_);
    mapped_ptr_ = nullptr;
    mapped_size_ = 0;
    is_valid_ = false;
  }
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

// SharedMemoryBuffer implementation
#ifdef __APPLE__
#include <sys/shm.h>
#define SHM_NAME_PREFIX "/tmp/"
#else
#include <sys/shm.h>
#endif

std::shared_ptr<SharedMemoryBuffer> SharedMemoryBuffer::create(
    const std::string& name, size_t size_bytes) {
#ifdef __APPLE__
  // macOS uses different shm naming
  std::string shm_name = "/tmp/shm_" + name;
#else
  std::string shm_name = "/" + name;
#endif

  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (shm_fd < 0) {
    if (errno == EEXIST) {
      throw std::runtime_error("Shared memory segment already exists: " + name);
    }
    throw std::runtime_error("Failed to create shared memory: " + name);
  }

  // Set size
  if (ftruncate(shm_fd, size_bytes) < 0) {
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error("Failed to set shared memory size");
  }

  // Map
  void* ptr =
      mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error("Failed to map shared memory");
  }

  return std::shared_ptr<SharedMemoryBuffer>(
      new SharedMemoryBuffer(name, ptr, size_bytes, shm_fd));
}

std::shared_ptr<SharedMemoryBuffer> SharedMemoryBuffer::attach(
    const std::string& name) {
#ifdef __APPLE__
  std::string shm_name = "/tmp/shm_" + name;
#else
  std::string shm_name = "/" + name;
#endif

  int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (shm_fd < 0) {
    throw std::runtime_error("Failed to open shared memory: " + name);
  }

  // Get size
  struct stat st;
  if (fstat(shm_fd, &st) < 0) {
    close(shm_fd);
    throw std::runtime_error("Failed to stat shared memory");
  }
  size_t size_bytes = static_cast<size_t>(st.st_size);

  // Map
  void* ptr =
      mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd);
    throw std::runtime_error("Failed to map shared memory");
  }

  return std::shared_ptr<SharedMemoryBuffer>(
      new SharedMemoryBuffer(name, ptr, size_bytes, shm_fd));
}

void SharedMemoryBuffer::destroy(const std::string& name) {
#ifdef __APPLE__
  std::string shm_name = "/tmp/shm_" + name;
#else
  std::string shm_name = "/" + name;
#endif
  shm_unlink(shm_name.c_str());
}

SharedMemoryBuffer::SharedMemoryBuffer(const std::string& name, void* ptr,
                                       size_t size_bytes, int shm_fd)
    : name_(name),
      shared_ptr_(ptr),
      size_bytes_(size_bytes),
      shm_fd_(shm_fd),
      owns_(false) {}

SharedMemoryBuffer::~SharedMemoryBuffer() { detach(); }

void SharedMemoryBuffer::detach() {
  if (shared_ptr_) {
    munmap(shared_ptr_, size_bytes_);
    shared_ptr_ = nullptr;
  }
  if (shm_fd_ >= 0) {
    close(shm_fd_);
    shm_fd_ = -1;
  }
}

}  // namespace dragon_tensor
