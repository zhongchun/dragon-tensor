#pragma once

#include <string>

namespace dragon_tensor {

// Logging level enumeration
enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, NONE = 4 };

// Set logging level
void set_log_level(LogLevel level);

// Get current logging level
LogLevel get_log_level();

// Logging functions
void log_debug(const std::string& message);
void log_info(const std::string& message);
void log_warning(const std::string& message);
void log_error(const std::string& message);

}  // namespace dragon_tensor
