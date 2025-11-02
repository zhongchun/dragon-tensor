#include <memory>
#include <string_view>

#include "dragon_tensor/backend.h"
#include "dragon_tensor/backends/arrow_backend.h"
#include "dragon_tensor/backends/memory_backend.h"
#include "dragon_tensor/backends/mmap_backend.h"
#include "dragon_tensor/backends/sharedmem_backend.h"

namespace dragon_tensor {

std::shared_ptr<Backend> create_memory_backend() {
  return std::make_shared<MemoryBackend>();
}

std::shared_ptr<Backend> create_mmap_backend(std::string_view path,
                                             bool read_only) {
  return std::make_shared<MMapBackend>(path, read_only);
}

std::shared_ptr<Backend> create_shared_memory_backend(std::string_view name) {
  return std::make_shared<SharedMemoryBackend>(name);
}

std::shared_ptr<Backend> create_arrow_backend(std::string_view path,
                                               bool read_only) {
  return std::make_shared<ArrowBackend>(path, read_only);
}

}  // namespace dragon_tensor
