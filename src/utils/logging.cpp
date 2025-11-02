#include "dragon_tensor/utils/logging.h"

#include <iostream>

namespace dragon_tensor {

namespace {
LogLevel g_log_level = LogLevel::WARNING;
}

void set_log_level(LogLevel level) { g_log_level = level; }

LogLevel get_log_level() { return g_log_level; }

void log_debug(const std::string& message) {
  if (g_log_level <= LogLevel::DEBUG) {
    std::cout << "[DEBUG] " << message << std::endl;
  }
}

void log_info(const std::string& message) {
  if (g_log_level <= LogLevel::INFO) {
    std::cout << "[INFO] " << message << std::endl;
  }
}

void log_warning(const std::string& message) {
  if (g_log_level <= LogLevel::WARNING) {
    std::cerr << "[WARNING] " << message << std::endl;
  }
}

void log_error(const std::string& message) {
  if (g_log_level <= LogLevel::ERROR) {
    std::cerr << "[ERROR] " << message << std::endl;
  }
}

}  // namespace dragon_tensor
