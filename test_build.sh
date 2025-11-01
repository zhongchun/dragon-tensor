#!/bin/bash
# Test script for Dragon Tensor build

echo "=== Testing Dragon Tensor Build ==="
echo ""

# Test 1: C++ Example
echo "1. Testing C++ example..."
if ./build/examples/example_basic > /dev/null 2>&1; then
    echo "   ✓ C++ example runs successfully"
    ./build/examples/example_basic | head -10
else
    echo "   ✗ C++ example failed"
    exit 1
fi

echo ""
echo "2. Testing Python module..."

# Create a symlink for easier import
cd build
if [ ! -e dragon_tensor.so ] && [ -f dragon_tensor*.dylib ]; then
    ln -sf dragon_tensor*.dylib dragon_tensor.so 2>/dev/null || cp dragon_tensor*.dylib dragon_tensor.so
fi
cd ..

# Test Python import
python3 << 'PYEOF'
import sys
import os
sys.path.insert(0, './build')

try:
    import dragon_tensor as dt
    import numpy as np
    
    print("   ✓ Python module imported successfully")
    
    # Test basic operations
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    tensor = dt.from_numpy_double(arr)
    
    assert abs(tensor.sum() - 15.0) < 1e-6, "Sum test failed"
    assert abs(tensor.mean() - 3.0) < 1e-6, "Mean test failed"
    print("   ✓ Basic operations work")
    
    # Test financial operations
    prices = np.array([100.0, 102.0, 101.0, 105.0, 108.0], dtype=np.float64)
    price_tensor = dt.from_numpy_double(prices)
    returns = price_tensor.returns()
    assert len(returns.to_numpy()) == 4, "Returns length incorrect"
    print("   ✓ Financial operations work")
    
    print("   ✓ All Python tests passed!")
    
except Exception as e:
    print(f"   ✗ Python test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

echo ""
echo "=== All tests passed! ==="
