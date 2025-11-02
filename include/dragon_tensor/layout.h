#pragma once

namespace dragon_tensor {

// Memory layout enumeration
enum class Layout { RowMajor, ColumnMajor };

// Helper functions for layout
inline const char* layout_to_string(Layout layout) {
  return layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor";
}

}  // namespace dragon_tensor

