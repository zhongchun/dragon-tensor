#include <dragon_tensor/tensor.h>

#include <iostream>
#include <vector>

using namespace dragon_tensor;

int main() {
  std::cout << "=== Dragon Tensor C++ Example ===\n\n";

  // Create a price series
  std::vector<double> prices = {100.0, 102.0, 101.0, 105.0, 108.0, 110.0};
  TensorDouble price_tensor({prices.size()}, prices);

  std::cout << "Price series: ";
  for (size_t i = 0; i < price_tensor.size(); ++i) {
    std::cout << price_tensor[i] << " ";
  }
  std::cout << "\n\n";

  // Calculate returns
  auto returns = price_tensor.returns();
  std::cout << "Returns: ";
  for (size_t i = 0; i < returns.size(); ++i) {
    std::cout << returns[i] << " ";
  }
  std::cout << "\n\n";

  // Statistical operations
  std::cout << "Mean: " << price_tensor.mean() << "\n";
  std::cout << "Std: " << price_tensor.std() << "\n";
  std::cout << "Max: " << price_tensor.max() << "\n";
  std::cout << "Min: " << price_tensor.min() << "\n\n";

  // Rolling statistics
  auto rolling_mean = price_tensor.rolling_mean(3);
  std::cout << "Rolling Mean (window=3): ";
  for (size_t i = 0; i < rolling_mean.size(); ++i) {
    std::cout << rolling_mean[i] << " ";
  }
  std::cout << "\n";

  auto rolling_std = price_tensor.rolling_std(3);
  std::cout << "Rolling Std (window=3): ";
  for (size_t i = 0; i < rolling_std.size(); ++i) {
    std::cout << rolling_std[i] << " ";
  }
  std::cout << "\n\n";

  // Correlation example - need matching sizes
  std::vector<double> prices2 = {50.0, 51.0, 50.5, 52.5, 54.0, 55.0};
  TensorDouble price_tensor2({prices2.size()}, prices2);

  auto corr = price_tensor.correlation(price_tensor2);
  std::cout << "Correlation: " << corr[0] << "\n";

  return 0;
}
