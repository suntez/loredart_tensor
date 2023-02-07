import '../tensors/num_tensor.dart';
import '/src/ops/other_ops.dart' show reshape;
import '/src/utils/dtype_utils.dart';
import '/src/tensors/tensor.dart';
import 'basic_ops.dart' show broadcastShapes;

// The type of the comparison operation
enum ComparisonType {equal, notEqual, greater, greaterEqual, less, lessEqual}

/// Compare [x] and [y] according to the comparison [type]
bool compare(num x, num y, ComparisonType type) {
  if (type == ComparisonType.equal) {
    return x == y;
  } else if (type == ComparisonType.notEqual) {
    return x != y;
  } else if (type == ComparisonType.greater) {
    return x > y;
  } else if (type == ComparisonType.greaterEqual) {
    return x >= y;
  } else if (type == ComparisonType.less) {
    return x < y;
  } else if (type == ComparisonType.lessEqual) {
    return x <= y;
  } else {
    return false;
  }
}

/// Compare [x] and [other] element-wise according to the comparison [type]
Tensor compareTensors(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  if (x.dType != other.dType) {
    throw ArgumentError('Tensors to compare must be of the same DType, but received ${x.dType} and ${other.dType}');
  }

   if (other.shape.size == 1) {
    if (other.rank > x.rank) {
      x = reshape(x, [...List.filled(other.rank-x.rank, 1), ...x.shape.list]) as NumericTensor;
    }
    return compareWithScalar(x, other.buffer[0], type: type);
  } else if (x.shape.size == 1) {
    if (x.rank > other.rank) {
      x = reshape(other, [...List.filled(x.rank-other.rank, 1), ...other.shape.list]) as NumericTensor;
    }
    return compareWithScalar(other, x.buffer[0], type: type);
  } else if (x.shape.compatibleWith(other.shape)) {
    return compareWithCompShapes(x, other, type: type);
  } else if (x.shape.equalWithLastDims(other.shape)) {
    return compareWithLastDims(x.rank > other.rank ? x : other, x.rank > other.rank ? other : x, type: type);
  } else if (other.shape.equalWithLastDims(x.shape)) {
    return compareWithLastDims(other, x, type: type);
  } else {
    throw ArgumentError('Tensors must be with compilable shapes, but received ${x.shape} and ${other.shape}');
  }
}
/// Compare elements of [x] to the [scalar] according to the comparison [type]
Tensor compareWithScalar(NumericTensor x, num scalar, {required ComparisonType type}) {
  List buffer = emptyBuffer(x.dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if(compare(x.buffer[i], scalar, type)) {
      buffer[i] = x.dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromTypedDataList(buffer, x.shape.list, dType: x.dType);
}

/// Compare two numeric tensors [x] and [other] element-wise according to the comparison [type]
/// 
/// Tensors are assumed to be of the same shape
Tensor compareWithEqualShapes(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  final dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if(compare(x.buffer[i], other.buffer[i], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
}

/// Compare two numeric tensors [x] and [other] element-wise according to the comparison [type]
/// 
/// Tensors are assumed to have compatible shapes
Tensor compareWithCompShapes(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  final dType = dTypeDecision(x.dType, other.dType);
  final List<int> shape = broadcastShapes(x.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (x.shape[i] == 1 ? 0 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
  final List<int> cumProdOther = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (other.shape[i] == 1 ? 0 : other.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));

  List<int> currentIndices = List<int>.filled(shape.length, 0);
  List buffer = emptyBuffer(dType, length);
  int tIndex = 0;
  int otherIndex = 0;
  for (int i = 0; i < length; i += 1) {
    tIndex = 0;
    otherIndex = 0;
    int index = i;
      for (int j = shape.length - 1; j >= 0; j -= 1) {
        currentIndices[j] = index % shape[j];
        index = index ~/ shape[j];
    }
    for (int k = 0; k < shape.length; k += 1) {
      tIndex += x.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }
    if (compare(x.buffer[tIndex], other.buffer[otherIndex], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromTypedDataList(buffer, shape, dType: dType);
}

/// Compare two numeric tensors [x] and [other] element-wise according to the comparison [type]
/// 
/// Tensors are assumed to have compatible last dims of shape
Tensor compareWithLastDims(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  if (other.shape.size > x.shape.size) {
    throw ArgumentError('Incorrect arguments order: expect to have x with bigger size than other');
  }
  final dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  final int residualSize = x.shape.list.sublist(0, x.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      if (compare(x.buffer[b*matchSize + i], other.buffer[i], type)) {
        buffer[b*matchSize + i] = dType.isInt ? 1 : 1.0;
      }
    }
  }
  return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
}

/// If can - converts [y] to the [NumericTensor], otherwise throws an ArgumentError
NumericTensor convertToNumericTensor(dynamic y) {
  if (y is NumericTensor) {
    return y;
  }
  else if (y is num) {
    return Tensor.constant([y], shape: [1]) as NumericTensor;
  } else {
    throw ArgumentError('Expected NumericalTensor or a num, but received ${y.runtimeType}', 'y');
  }
}

/// Returns the "truth" value of [x] == [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(equal(x, y)); // <Tensor(shape: [2, 2], values: [[1.0, 0.0] [0.0, 1.0]], dType: float32)>
/// 
/// print(equal(x, 2.0)); // <Tensor(shape: [2, 2], values: [[0.0, 1.0] [0.0, 0.0]], dType: float32)>
/// 
/// // but this won't work
/// print(equal(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor equal(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.equal);
}

/// Returns the "truth" value of [x] > [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(greater(x, y)); // <Tensor(shape: [2, 2], values: [[0.0, 0.0] [1.0, 0.0]], dType: float32)>
/// 
/// print(greater(x, 2.0)); // <Tensor(shape: [2, 2], values: [[0.0, 0.0] [1.0, 1.0]], dType: float32)>
/// 
/// // but this won't work
/// print(greater(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor greater(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greater);
}

/// Returns the "truth" value of [x] >= [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(greaterEqual(x, y)); // <Tensor(shape: [2, 2], values: [[1.0, 0.0] [1.0, 1.0]], dType: float32)>
/// 
/// print(greaterEqual(x, 2.0)); // <Tensor(shape: [2, 2], values: [[0.0, 1.0] [1.0, 1.0]], dType: float32)>
/// 
/// // but this won't work
/// print(greaterEqual(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor greaterEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greaterEqual);
}

/// Returns the "truth" value of [x] < [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(less(x, y)); // <Tensor(shape: [2, 2], values: [[0.0, 1.0] [0.0, 0.0]], dType: float32)>
/// 
/// print(less(x, 2.0)); // <Tensor(shape: [2, 2], values: [[1.0, 0.0] [0.0, 0.0]], dType: float32)>
/// 
/// // but this won't work
/// print(less(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor less(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.less);
}

/// Returns the "truth" value of [x] <= [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(lessEqual(x, y)); // <Tensor(shape: [2, 2], values: [[1.0, 1.0] [1.0, 0.0]], dType: float32)>
/// 
/// print(lessEqual(x, 2.0)); // <Tensor(shape: [2, 2], values: [[1.0, 1.0] [0.0, 0.0]], dType: float32)>
/// 
/// // but this won't work
/// print(lessEqual(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor lessEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.lessEqual);
}

/// Returns the "truth" value of [x] <= [y] element-wise, represented as 1 for `true` and 0 for `false`.
/// 
/// Operand [y] can be a `num` or a `NumericalTensor` with broadcastable shape with [x.shape],
/// otherwise will throw an ArgumentError. 
/// In any case, [x] and [y] must be of the same [DType] (or [Type] if [y] is a num).
/// 
/// Returns a [Tensor] of the same [DType] as [x] and [y], and with broadcasted shape.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant([1.0, 2.0, 3.0, 4.0], shape: [2,2]);
/// final y = Tensor.constant([1.0, 4.0]);
/// print(notEqual(x, y)); // <Tensor(shape: [2, 2], values: [[0.0, 1.0] [1.0, 0.0]], dType: float32)>
/// 
/// print(notEqual(x, 2.0)); // <Tensor(shape: [2, 2], values: [[1.0, 0.0] [1.0, 1.0]], dType: float32)>
/// 
/// // but this won't work
/// print(notEqual(x, 1)) // Invalid argument(s): Tensors to compare.... 
/// ```
Tensor notEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to be NumericTensor, but received tensor of ${x.dType}', 'x');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.notEqual);
}
