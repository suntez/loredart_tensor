/// A Dart-pure package for manipulation with tensors (multidimensional arrays of data) inspired by the TensorFlow API.
library loredart_tensor;

export 'src/tensors/tensor_shape.dart';
export 'src/tensors/tensor.dart';
export 'src/tensors/num_tensor.dart';
export 'src/ops/math_ops.dart';
export 'src/ops/reduce_ops.dart';
export 'src/ops/linalg.dart';
export 'src/ops/other_ops.dart';
export 'src/ops/random.dart';
export 'src/utils/dtype_utils.dart' show emptyBuffer;
export 'src/ops/comparison_ops.dart' show equal, notEqual, less, lessEqual, greater, greaterEqual;
export 'src/serialization/tensor_serialization.dart';