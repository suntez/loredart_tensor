import 'dart:math' show Random;
import 'package:loredart_tensor/loredart_tensor.dart';

import '../tensors/tensor.dart';
import '../utils/random_utils.dart';

/// Outputs [Tensor] of given [shape] with random values from a uniform distribution.
/// 
/// The generated values follow a uniform distribution in the range [[min], [maxval]),
/// the lower bound is included, while the upper bound - excluded.
/// 
/// The default range is [0, 1) (in such case for int based [dType] will returns zeros).
/// 
/// If [min] > [max], will swap [min] and [max].
/// 
/// Throws an ArgumentError if given [DType] is non-numeric.
/// 
/// Example:
/// ```dart
/// Tensor x = uniform([2,2], min: -1, max: 2, seed: 1);
/// print(x);
/// // <Tensor(shape: [2, 2], values:
/// //  [[..., ...]
/// //  [..., ...]], dType: float32)>
///
/// Tensor y = uniform([1, 3], min: 3, max: 4, dType: int32);
/// print(y);
/// // <Tensor(shape: [1, 3], values:
/// //  [[3, 3, 3]], dType: int32)>
/// ```
Tensor uniform(List<int> shape, {num min = 0.0, num max = 1.0, DType dType = DType.float32, int? seed}) {
  if (dType.isDouble) {
    final Random random = Random(seed);
    min = min.toDouble();
    max = max.toDouble();
    if (min > max) {
      var swapTemp = min;
      min = max;
      max = swapTemp;
    }
    List<double> values = List.generate(shape.reduce((e1, e2) => e1*e2), (_) => (min + random.nextDouble() * (max-min)));
    return Tensor.constant(values, shape: shape, dType: dType);
  } else if (dType.isInt) {
    final Random random = Random(seed);
    min = min.toInt();
    max = max.toInt();
    if (min > max) {
      var swapTemp = min;
      min = max;
      max = swapTemp;
    }
    List<int> values = List.generate(shape.reduce((e1, e2) => e1*e2), (_) => (min + random.nextInt((max-min).toInt())).toInt());
    return Tensor.constant(values, shape: shape, dType: dType);
  } else {
    throw ArgumentError('DType $dType is not supported for uniform random generation', 'dType');
  }
}

/// Outputs [Tensor] of given [shape] with random values from a normal distribution with mean [mean] and standard deviation [std].
/// 
/// Throws an ArgumentError if [dType] is not [DType.float32] or [DType.float64].
/// 
/// Example:
/// ```dart
/// Tensor x = normal([3,3], std: 2.0);
/// print(x);
/// // <Tensor(shape: [3, 3], values:
/// // [[...]], dType: float32)>
/// ```
Tensor normal(List<int> shape, {num mean = 0.0, num std = 1.0, DType dType = DType.float32, int? seed}) {
  if (dType.isDouble) {
    List<double> values = generateNormallyDistList(shape.reduce((e1,e2) => e1*e2), mean, std, seed: seed);
    return Tensor.constant(values, shape: shape, dType: dType);
  } else {
    throw ArgumentError('DType $dType is not supported for normal random generation', 'dType');
  }
}

/// Outputs [Tensor] of given [shape] with random values from a truncated normal distribution.
/// 
/// The values are drawn from a normal distribution with specified [mean] and [std],
/// discarding and re-drawing any samples that are more than two standard deviations from the [mean].
/// 
/// Throws an ArgumentError if [dType] is not [DType.float32] or [DType.float64].
/// 
/// Example:
/// ```dart
/// Tensor x = truncatedNormal([3,3], mean: 1.0, std: 2.0);
/// print(x);
/// // <Tensor(shape: [3, 3], values:
/// // [[...]], dType: float32)>
/// ```
Tensor truncatedNormal(List<int> shape, {num mean = 0.0, num std = 1.0, DType dType = DType.float32, int? seed}) {
  if (dType.isDouble) {
    List<double> values = generateTruncatedNormallyDistList(shape.reduce((e1,e2) => e1*e2), mean, std, seed: seed);
    return Tensor.constant(values, shape: shape, dType: dType);
  } else {
    throw ArgumentError('DType $dType is not supported for normal random generation', 'dType');
  }
}