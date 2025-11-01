#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <dragon_tensor/tensor.h>
#include <vector>
#include <memory>

namespace py = pybind11;
using namespace dragon_tensor;

// Helper function to convert numpy array to Tensor (zero-copy when possible)
template<typename T>
Tensor<T> numpy_to_tensor(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim == 0) {
        throw std::runtime_error("Cannot convert 0-dimensional array to tensor");
    }
    
    std::vector<size_t> shape;
    for (py::ssize_t i = 0; i < buf.ndim; ++i) {
        shape.push_back(static_cast<size_t>(buf.shape[i]));
    }
    
    // Check if array is C-contiguous for optimized copy
    bool is_contiguous = (buf.format == py::format_descriptor<T>::format() && 
                         buf.size > 0 && 
                         buf.strides.empty());  // Contiguous if no strides or all 1
    
    // Tensor uses std::vector which owns memory, so we always copy
    // But we optimize by checking contiguity to avoid extra allocations
    std::vector<T> data;
    if (is_contiguous && buf.size > 0) {
        T* ptr = static_cast<T*>(buf.ptr);
        data.assign(ptr, ptr + buf.size);
    } else {
        // Non-contiguous: need to copy
        data.resize(buf.size);
        std::memcpy(data.data(), buf.ptr, buf.size * sizeof(T));
    }
    
    return Tensor<T>(shape, std::move(data));
}

// Helper function to convert Tensor to numpy array (zero-copy)
// This version uses a Python object to keep the tensor alive
template<typename T>
py::array_t<T> tensor_to_numpy_zero_copy(const Tensor<T>& tensor, py::object owner) {
    std::vector<size_t> shape = tensor.shape();
    std::vector<py::ssize_t> py_shape;
    std::vector<py::ssize_t> py_strides;
    
    for (size_t s : shape) {
        py_shape.push_back(static_cast<py::ssize_t>(s));
    }
    
    // Calculate strides for C-contiguous layout (row-major)
    py::ssize_t stride = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        py_strides.insert(py_strides.begin(), stride);
        stride *= static_cast<py::ssize_t>(shape[i]);
    }
    
    // Zero-copy: create array that shares memory with Tensor
    // The 'owner' Python object keeps the tensor alive
    const T* data_ptr = tensor.raw_data();
    
    // Create a capsule to hold the owner object and keep it alive
    // Use a unique_ptr to manage the owner's lifetime
    struct OwnerHolder {
        py::object owner;
        OwnerHolder(py::object o) : owner(std::move(o)) {}
    };
    auto holder = std::make_unique<OwnerHolder>(owner);
    
    py::capsule capsule(holder.get(), [](void* p) {
        delete static_cast<OwnerHolder*>(p);
    });
    
    // Transfer ownership to capsule
    holder.release();
    
    // Create array with zero-copy view of Tensor's data
    return py::array_t<T>(
        py_shape,
        py_strides,
        const_cast<T*>(data_ptr),  // NumPy array shares this memory (zero-copy)
        capsule  // Capsule keeps owner alive, ensuring tensor remains valid
    );
}

// Legacy function for backward compatibility (creates a copy to manage lifetime)
template<typename T>
py::array_t<T> tensor_to_numpy(const Tensor<T>& tensor) {
    std::vector<size_t> shape = tensor.shape();
    std::vector<py::ssize_t> py_shape;
    std::vector<py::ssize_t> py_strides;
    
    for (size_t s : shape) {
        py_shape.push_back(static_cast<py::ssize_t>(s));
    }
    
    // Calculate strides for C-contiguous layout (row-major)
    py::ssize_t stride = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        py_strides.insert(py_strides.begin(), stride);
        stride *= static_cast<py::ssize_t>(shape[i]);
    }
    
    // Zero-copy: create array that shares memory with Tensor
    // We need to keep the Tensor alive while NumPy array exists
    // Create a shared_ptr copy of the tensor to manage lifetime
    auto tensor_holder = std::make_shared<Tensor<T>>(tensor);
    
    // Get data pointer from the held tensor (not the original, to ensure it stays valid)
    T* data_ptr = tensor_holder->raw_data();
    
    // Create a capsule to keep the tensor alive
    // The capsule stores a shared_ptr, which ensures the tensor (and its data)
    // stays alive as long as the NumPy array exists
    py::capsule owner(new std::shared_ptr<Tensor<T>>(tensor_holder), 
                      [](void* p) {
                          delete static_cast<std::shared_ptr<Tensor<T>>*>(p);
                      });
    
    // Create array with zero-copy view of Tensor's data
    // The data_ptr points to memory owned by tensor_holder
    // The capsule ensures tensor_holder lives as long as the array
    return py::array_t<T>(
        py_shape,
        py_strides,
        data_ptr,  // NumPy array shares this memory (zero-copy)
        owner  // Capsule keeps tensor_holder alive, ensuring data_ptr remains valid
    );
}

// Template to bind Tensor operations
template<typename T>
void bind_tensor_operations(py::module& m, const std::string& name_suffix) {
    std::string class_name = "Tensor" + name_suffix;
    
    py::class_<Tensor<T>>(m, class_name.c_str())
        // Constructors
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<T>&>())
        .def(py::init<const std::vector<size_t>&, T>())
        .def(py::init([](py::array_t<T> arr) {
            return numpy_to_tensor(arr);
        }), "Constructor from numpy array")
        
        // Shape operations
        .def("shape", &Tensor<T>::shape)
        .def("ndim", &Tensor<T>::ndim)
        .def("size", &Tensor<T>::size)
        .def("empty", &Tensor<T>::empty)
        
        // Reshape and flatten
        .def("reshape", &Tensor<T>::reshape)
        .def("flatten", &Tensor<T>::flatten)
        
        // Element access
        .def("__getitem__", [](const Tensor<T>& t, size_t index) {
            return t[index];
        })
        .def("__setitem__", [](Tensor<T>& t, size_t index, T value) {
            t[index] = value;
        })
        .def("at", py::overload_cast<size_t>(&Tensor<T>::at, py::const_))
        .def("at", py::overload_cast<const std::vector<size_t>&>(&Tensor<T>::at, py::const_))
        
        // Data access
        .def("data", &Tensor<T>::data)
        .def("to_numpy", [](py::object self) {
            // Zero-copy: get the C++ tensor from Python object
            const Tensor<T>& t = self.cast<const Tensor<T>&>();
            return tensor_to_numpy_zero_copy(t, self);
        }, "Convert to numpy array (zero-copy)")
        
        // Arithmetic operations
        .def("__add__", [](const Tensor<T>& a, const Tensor<T>& b) {
            return a + b;
        })
        .def("__add__", [](const Tensor<T>& a, T scalar) {
            return a + scalar;
        })
        .def("__radd__", [](const Tensor<T>& a, T scalar) {
            return a + scalar;
        })
        .def("__sub__", [](const Tensor<T>& a, const Tensor<T>& b) {
            return a - b;
        })
        .def("__sub__", [](const Tensor<T>& a, T scalar) {
            return a - scalar;
        })
        .def("__rsub__", [](const Tensor<T>& a, T scalar) {
            Tensor<T> result = a;
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = scalar - result[i];
            }
            return result;
        })
        .def("__mul__", [](const Tensor<T>& a, const Tensor<T>& b) {
            return a * b;
        })
        .def("__mul__", [](const Tensor<T>& a, T scalar) {
            return a * scalar;
        })
        .def("__rmul__", [](const Tensor<T>& a, T scalar) {
            return a * scalar;
        })
        .def("__truediv__", [](const Tensor<T>& a, const Tensor<T>& b) {
            return a / b;
        })
        .def("__truediv__", [](const Tensor<T>& a, T scalar) {
            return a / scalar;
        })
        .def("__rtruediv__", [](const Tensor<T>& a, T scalar) {
            Tensor<T> result = a;
            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i] == T(0)) {
                    throw std::runtime_error("Division by zero");
                }
                result[i] = scalar / result[i];
            }
            return result;
        })
        
        // In-place operations
        .def("__iadd__", [](Tensor<T>& a, const Tensor<T>& b) -> Tensor<T>& {
            a += b;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__iadd__", [](Tensor<T>& a, T scalar) -> Tensor<T>& {
            a += scalar;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor<T>& a, const Tensor<T>& b) -> Tensor<T>& {
            a -= b;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__isub__", [](Tensor<T>& a, T scalar) -> Tensor<T>& {
            a -= scalar;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor<T>& a, const Tensor<T>& b) -> Tensor<T>& {
            a *= b;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__imul__", [](Tensor<T>& a, T scalar) -> Tensor<T>& {
            a *= scalar;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor<T>& a, const Tensor<T>& b) -> Tensor<T>& {
            a /= b;
            return a;
        }, py::return_value_policy::reference_internal)
        .def("__itruediv__", [](Tensor<T>& a, T scalar) -> Tensor<T>& {
            a /= scalar;
            return a;
        }, py::return_value_policy::reference_internal)
        
        // Comparison
        .def("__eq__", &Tensor<T>::operator==)
        .def("__ne__", &Tensor<T>::operator!=)
        
        // Mathematical operations
        .def("abs", &Tensor<T>::abs)
        .def("sqrt", &Tensor<T>::sqrt)
        .def("exp", &Tensor<T>::exp)
        .def("log", &Tensor<T>::log)
        .def("pow", &Tensor<T>::pow)
        
        // Statistical operations
        .def("sum", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::sum))
        .def("sum", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::sum))
        .def("mean", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::mean))
        .def("mean", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::mean))
        .def("max", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::max))
        .def("max", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::max))
        .def("min", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::min))
        .def("min", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::min))
        .def("std", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::std))
        .def("std", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::std))
        .def("var", static_cast<T (Tensor<T>::*)() const>(&Tensor<T>::var))
        .def("var", static_cast<Tensor<T> (Tensor<T>::*)(size_t) const>(&Tensor<T>::var))
        
        // Financial operations
        .def("returns", &Tensor<T>::returns)
        .def("rolling_mean", &Tensor<T>::rolling_mean)
        .def("rolling_std", &Tensor<T>::rolling_std)
        .def("rolling_sum", &Tensor<T>::rolling_sum)
        .def("rolling_max", &Tensor<T>::rolling_max)
        .def("rolling_min", &Tensor<T>::rolling_min)
        
        // Correlation and covariance
        .def("correlation", &Tensor<T>::correlation)
        .def("covariance", &Tensor<T>::covariance)
        
        // Slicing
        .def("slice", &Tensor<T>::slice)
        .def("slice_row", &Tensor<T>::slice_row)
        .def("slice_column", &Tensor<T>::slice_column)
        
        // Matrix operations
        .def("transpose", &Tensor<T>::transpose)
        .def("matmul", &Tensor<T>::matmul)
        
        // Copy
        .def("copy", &Tensor<T>::copy)
        
        // String representation
        .def("__repr__", [class_name](const Tensor<T>& t) {
            std::string repr = class_name + "(shape=";
            auto shape = t.shape();
            repr += "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                repr += std::to_string(shape[i]);
                if (i < shape.size() - 1) repr += ", ";
            }
            repr += "])";
            return repr;
        });
}

PYBIND11_MODULE(dragon_tensor, m) {
    m.doc() = "Dragon Tensor - High-performance tensor library for financial data analysis";
    
    // Export tensor classes
    bind_tensor_operations<float>(m, "Float");
    bind_tensor_operations<double>(m, "Double");
    bind_tensor_operations<int32_t>(m, "Int");
    bind_tensor_operations<int64_t>(m, "Long");
    
    // Convenience functions for creating tensors from numpy arrays
    m.def("from_numpy_float", [](py::array_t<float> arr) {
        return numpy_to_tensor(arr);
    }, "Create Tensor from numpy float array");
    
    m.def("from_numpy_double", [](py::array_t<double> arr) {
        return numpy_to_tensor(arr);
    }, "Create Tensor from numpy double array");
    
    m.def("from_numpy_int", [](py::array_t<int32_t> arr) {
        return numpy_to_tensor(arr);
    }, "Create Tensor from numpy int32 array");
    
    m.def("from_numpy_long", [](py::array_t<int64_t> arr) {
        return numpy_to_tensor(arr);
    }, "Create Tensor from numpy int64 array");
    
    // Helper function to convert pandas Series/DataFrame
    m.def("from_pandas_series", [](py::object series) -> py::object {
        try {
            py::array values = series.attr("values");
            auto dtype = values.dtype();
            
            if (dtype.is(py::dtype::of<float>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<float>>(values)));
            } else if (dtype.is(py::dtype::of<double>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<double>>(values)));
            } else if (dtype.is(py::dtype::of<int32_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int32_t>>(values)));
            } else if (dtype.is(py::dtype::of<int64_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int64_t>>(values)));
            } else {
                throw std::runtime_error("Unsupported pandas dtype");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to convert pandas Series: " + std::string(e.what()));
        }
    }, "Create Tensor from pandas Series");
    
    m.def("from_pandas_dataframe", [](py::object df) -> py::object {
        try {
            py::array values = df.attr("values");
            auto dtype = values.dtype();
            
            if (dtype.is(py::dtype::of<float>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<float>>(values)));
            } else if (dtype.is(py::dtype::of<double>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<double>>(values)));
            } else if (dtype.is(py::dtype::of<int32_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int32_t>>(values)));
            } else if (dtype.is(py::dtype::of<int64_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int64_t>>(values)));
            } else {
                throw std::runtime_error("Unsupported pandas dtype");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to convert pandas DataFrame: " + std::string(e.what()));
        }
    }, "Create Tensor from pandas DataFrame");
    
    // Helper function to convert torch tensor (zero-copy via NumPy)
    m.def("from_torch", [](py::object torch_tensor) -> py::object {
        try {
            // PyTorch's .numpy() already returns a zero-copy view when possible
            // Convert to numpy first (this is zero-copy if tensor is on CPU and contiguous)
            py::array arr = torch_tensor.attr("detach")().attr("cpu")().attr("numpy")();
            auto dtype = arr.dtype();
            
            // Now convert numpy to tensor (will be efficient for contiguous arrays)
            if (dtype.is(py::dtype::of<float>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<float>>(arr)));
            } else if (dtype.is(py::dtype::of<double>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<double>>(arr)));
            } else if (dtype.is(py::dtype::of<int32_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int32_t>>(arr)));
            } else if (dtype.is(py::dtype::of<int64_t>())) {
                return py::cast(numpy_to_tensor(py::cast<py::array_t<int64_t>>(arr)));
            } else {
                throw std::runtime_error("Unsupported torch dtype");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to convert torch tensor: " + std::string(e.what()));
        }
    }, "Create Tensor from torch tensor (zero-copy when torch tensor is CPU and contiguous)");
}

