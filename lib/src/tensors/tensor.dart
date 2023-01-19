import 'dart:typed_data';

import '/src/utils/diagonal_utils.dart';
import '/src/utils/shape_utils.dart';

import '/src/math/basic_operations.dart';
import '/src/math/broadcastable_basic_operations.dart';
import '/src/utils/dtype_utils.dart';
import 'float32list_num_tensor.dart';
import 'float64list_num_tensor.dart';
import 'int32list_num_tensor.dart';
import 'int64list_num_tensor.dart';
import 'tensor_shape.dart';


/// The data types of the elements in a [Tensor].
enum DType { float32, float64, int32, int64, string, bool }

/// The data types of the elements in a [Tensor].
extension DTypeTypes on DType {
  /// True if [this] is integer-based type
  bool get isInt => this == DType.int32 || this == DType.int64;

  /// True if [this] is double-based type
  bool get isDouble => this == DType.float32 || this == DType.float64;

  /// True if [this] is num-based type
  bool get isNumeric => this != DType.bool && this != DType.string;
}

/// Default correspondence of Dart [Type]s and tensors' [DType]. 
const Map<Type, DType> defaultTypesToDTypes = {
  int: DType.int32,
  double: DType.float32,
  bool: DType.bool,
  String: DType.string
};

/// The representation of a multidimensional array of elements of a single type.
/// 
/// 
/// 
/// 
/// 
abstract class Tensor {
  /// The shape of the [Tensor].
  late final TensorShape shape;

  /// The data type of [Tensor]'s elements.
  late final DType dType;

  /// Number of dims in the [shape].
  /// 
  /// [rank] is a number of indices needed to get single element of a [Tensor].
  int get rank => shape.rank;

  /// Creates [Tensor] from [values].
  /// 
  /// [values] may be list of [num] or have a multidimensional structure. Non-numerical lists aren't supported yet.
  /// 
  /// If [shape] is specified, tries to reshape elements to match, otherwise inherits shape from [values].
  /// If number of elements of [values] won't be equal to [shape.size] throws an [ArgumentError].
  /// 
  /// If [dType] is specified, casts elements of [value] to meet the type, otherwise will inherit [dType] according to `defaultTypesToDTypes`.
  /// 
  /// If [values] contains non-numerical elements or nested lists have non-equal length will throw an [ArgumentError].
  /// 
  /// Examples:
  /// ```dart
  /// Tensor x = Tensor.constant([1,2,3,4]); // vector
  /// print(x);
  /// // <Tensor(shape: [4], values:
  /// // [[1, 2, 3, 4]], dType: int32)>
  /// 
  /// Tensor y = Tensor.constant([1,2,3,4], dType: DType.float32); // vector of concrete dType
  /// print(y);
  /// // <Tensor(shape: [4], values:
  /// // [[1.0, 2.0, 3.0, 4.0]], dType: float32)>
  ///
  /// Tensor u = Tensor.constant([[1,2], [3,4]]); // nested list as matrix
  /// print(u);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[1, 2]
  /// //  [3, 4]], dType: int32)>
  ///
  /// Tensor t = Tensor.constant([1,2,3,4,5,6], shape: [3,2]); // from vector to any tensor
  /// print(t);
  /// // <Tensor(shape: [3, 2], values:
  /// // [[1, 2]
  /// // [3, 4]
  /// // [5, 6]], dType: int32)>
  ///
  /// Tensor s = Tensor.constant([[1,2,3], [4,5,6]], shape: [6]); // reshaping nested lists
  ///  print(s);
  /// // <Tensor(shape: [6], values:
  /// // [[1, 2, 3, 4, 5, 6]], dType: int32)>
  /// 
  /// ```
  factory Tensor.constant(List<dynamic> values, {List<int>? shape, DType? dType}) {
    final flatten = flattenList(values);
    shape ??= extractDimsFromNestedValues(values);
    dType ??= defaultTypesToDTypes[flatten[0].runtimeType];
    if (shape.reduce((e1,e2) => e1*e2) == flatten.length) {
      if (dType == DType.float32) {
        return Float32NumericTensor.fromBuffer(
            Float32List.fromList(List.generate(flatten.length, (i) => flatten[i].toDouble(), growable: false)), shape);
      } else if (dType == DType.float64) {
        return Float64NumericTensor.fromBuffer(
            Float64List.fromList(List.generate(flatten.length, (i) => flatten[i].toDouble(), growable: false)), shape);
      } else if (dType == DType.int32) {
        return Int32NumericTensor.fromBuffer(
            Int32List.fromList(List.generate(flatten.length, (i) => flatten[i].toInt(), growable: false)), shape);
      } else if (dType == DType.int64) {
        return Int64NumericTensor.fromBuffer(
            Int64List.fromList(List.generate(flatten.length, (i) => flatten[i].toInt(), growable: false)), shape);
      } else {
        throw ArgumentError(
            'DType $dType is not supported for Tensor.constant factory', 'dType');
      }
    } else {
      throw ArgumentError('Cannot construct Tensor with ${flatten.length} elements with shape $shape', 'shape');
    }
  }

  /// Creates a [Tensor] of type [dType] with [shape] and all elements set to zero.
  /// 
  /// Throws an [ArgumentError] if [dType] is non-numeric.
  /// 
  /// Example:
  ///  ```dart
  /// Tensor x = Tensor.zeros([2,2], dType: int32);
  /// print(x);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[0, 0]
  /// // [0, 0]], dType: int32)>
  /// ```
  factory Tensor.zeros(List<int> shape, {DType dType = DType.float32}) {
    if (dType == DType.float32) {
      return Float32NumericTensor.zeros(shape);
    } else if (dType == DType.float64) {
      return Float64NumericTensor.zeros(shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.zeros(shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.zeros(shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.zeros factory', 'dType');
    }
  }

  factory Tensor.fromBuffer(List buffer, List<int> shape,
      {DType dType = DType.float32}) {
    if (buffer.length != shape.reduce((e1,e2)=>e1*e2)) {
      throw ArgumentError('Cannot allocate ${buffer.length} elements into shape $shape');
    }
    if (dType == DType.float32) {
      return Float32NumericTensor.fromBuffer(buffer as Float32List, shape);
    } else if (dType == DType.float64) {
      return Float64NumericTensor.fromBuffer(buffer as Float64List, shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.fromBuffer(buffer as Int32List, shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.fromBuffer(buffer as Int64List, shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.fromBuffer factory', 'dType');
    }
  }

  /// Creates an identity matrix (or batch of matrices) of type [dType].
  /// 
  /// Constructs square matrix of size [numRows]x[numRows] or rect matrix of size [numRows]x[numCols] if [numCols] is specified.
  /// 
  /// Returns [Tensor] of shape `[..batchSize, [numRows], [numCols]]` if [batchSize] is specified.
  /// 
  /// Throws an [ArgumentError] if [dType] is non-numeric.
  /// 
  /// Example:
  /// ```dart
  /// Tensor x = Tensor.eye(2); // square matrix
  /// print(x);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[1.0, 0.0]
  /// //  [0.0, 1.0]], dType: float32)>
  /// 
  /// Tensor y = Tensor.eye(2, numCols: 3, // rect matrix
  ///   dType: DType.int32);
  /// print(y);
  /// // <Tensor(shape: [2, 3], values:
  /// // [[1, 0, 0]
  /// //  [0, 1, 0]], dType: int32)>
  /// 
  /// Tensor u = Tensor.eye(2, batchShape: [2, 1]); // batched matrices
  /// print(u);
  /// // <Tensor(shape: [2, 1, 2, 2], values:
  /// // [[[[1.0 0.0]
  /// //   [0.0 1.0]]]
  /// // [[[1.0 0.0]
  /// //   [0.0 1.0]]]], dType: float32)>
  /// ```
  factory Tensor.eye(int numRows,
      {int? numCols,
      List<int>? batchShape,
      DType dType = DType.float32}) {
    if (dType == DType.float32) {
      return Float32NumericTensor.eye(
          numRows, numCols ?? numRows, batchShape ?? []);
    } else if (dType == DType.float64) {
      return Float64NumericTensor.eye(
          numRows, numCols ?? numRows, batchShape ?? []);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.eye(
          numRows, numCols ?? numRows, batchShape ?? []);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.eye(
          numRows, numCols ?? numRows, batchShape ?? []);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.eye factory', 'dType');
    }
  }

  /// Creates a [Tensor] of type [dType] with [shape] and all elements set to [value].
  /// 
  /// [Tensor.fill] will cast [value] according to the [dType].
  /// 
  /// Throws an [ArgumentError] if [dType] is not numeric.
  /// 
  /// Example:
  ///  ```dart
  /// Tensor x = Tensor.zeros([2,2], 13); // default dType is float32
  /// print(x);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[13.0, 13.0]
  /// // [13.0, 13.0]], dType: float32)>
  /// ```
  /// 
  /// Example with casting [value]:
  ///  ```dart
  /// Tensor y = Tensor.zeros([2,2], 13.9234, dType: int32);
  /// print(y);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[13, 13]
  /// // [13, 13]], dType: int32)>
  /// ```
  factory Tensor.fill(List<int> shape, dynamic value,
      {DType dType = DType.float32}) {
    if (dType == DType.float32) {
      return Float32NumericTensor.fill(shape, (value as num).toDouble());
    } else if (dType == DType.float64) {
      return Float64NumericTensor.fill(shape, (value as num).toDouble());
    } else if (dType == DType.int32) {
      return Int32NumericTensor.fill(shape, (value as num).toInt());
    } else if (dType == DType.int64) {
      return Int64NumericTensor.fill(shape, (value as num).toInt());
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.fill factory', 'dType');
    }
  }

  /// Creates a [Tensor] of type [dType] with [shape] and all elements set to one.
  /// 
  /// Throws an [ArgumentError] if [dType] is not numeric.
  /// 
  /// Example:
  ///  ```dart
  /// Tensor x = Tensor.ones([2,2], dType: float64);
  /// print(x);
  /// // <Tensor(shape: [2, 2], values:
  /// // [[1.0, 1.0]
  /// // [1.0, 1.0]], dType: float64)>
  /// ```
  factory Tensor.ones(List<int> shape, {DType dType = DType.float32}) {
    if (dType == DType.float32) {
      return Float32NumericTensor.ones(shape);
    } else if (dType == DType.float64) {
      return Float64NumericTensor.ones(shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.ones(shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.ones(shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.ones factory', 'dType');
    }
  }

  //TODO: API for diag
  factory Tensor.diag(List<dynamic> diagonal,
      {int offset = 0,
      int? numRows,
      int? numCols,
      DType dType = DType.float32}) {
    if (dType.isNumeric) {
      return createDiagTensor(diagonal, dType, offset, numRows, numCols);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.diag factory', 'dType');
    }
  }

  Tensor operator +(Object other);
  Tensor operator -(Object other);
  Tensor operator *(Object other);
  Tensor operator /(Object other);
  Tensor operator -();

  @override
  String toString() {
    return '<Tensor(shape: $shape, dtype: $dType)>';
  }

  String toStringShort() {
    return '<Tensor(shape: $shape, dtype: $dType)>';
  }
}

abstract class NumericTensor<L extends List> implements Tensor {
  late final L buffer;
  @override
  late final TensorShape shape;
  @override
  late final DType dType;

  @override
  int get rank => shape.rank;

  @override
  Tensor operator +(Object other) {
    if (other is num) {
      final resultDType = dTypeAndNumDecision(dType, other.runtimeType);
      return addScalar(this, other, resultDType);
    } else if (other is NumericTensor) {
      final resultDType = dTypeDecision(dType, other.dType);
      if (shape.equalWith(other.shape)) {
        return add(this, other, resultDType);
      } else if (shape.compatibleWith(other.shape)) {
        return addWithCompShape(this, other, resultDType);
      } else if (shape.equalWithLastDims(other.shape)) {
        return addWithLastDims(rank > other.rank ? this : other,
            rank < other.rank ? this : other, resultDType);
      } else if (other.shape.size == 1) {
        return addScalar(this, other.buffer[0], resultDType);
      } else {
        throw ArgumentError.value(other.shape, '',
            'Error shape: ${other.shape} is not compatible with $shape');
      }
    } else {
      throw ArgumentError(
          'Expected num or Tensor, but got ${other.runtimeType}',
          'loredart_tensor');
    }
  }

  @override
  Tensor operator -(Object other) {
    if (other is num) {
      final resultDType = dTypeAndNumDecision(dType, other.runtimeType);
      return subtractScalar(this, other, resultDType);
    } else if (other is NumericTensor) {
      final resultDType = dTypeDecision(dType, other.dType);
      if (shape.equalWith(other.shape)) {
        return subtract(this, other, resultDType);
      } else if (shape.compatibleWith(other.shape)) {
        return subtractWithCompShape(this, other, resultDType);
      } else if (shape.equalWithLastDims(other.shape)) {
        return subtractWithLastDims(
            rank > other.rank ? this : other,
            rank < other.rank ? this : other,
            resultDType);
      } else if (other.shape.size == 1) {
        return subtractScalar(this, other.buffer[0], resultDType);
      } else {
        throw ArgumentError.value(other.shape, '',
            'Error shape: ${other.shape} is not compatible with $shape');
      }
    } else {
      throw ArgumentError(
          'Expected num or Tensor, but got ${other.runtimeType}',
          'loredart_tensor');
    }
  }

  @override
  Tensor operator *(Object other) {
    if (other is num) {
      final resultDType = dTypeAndNumDecision(dType, other.runtimeType);
      return multiplyScalar(this, other, resultDType);
    } else if (other is NumericTensor) {
      final resultDType = dTypeDecision(dType, other.dType);
      if (shape.equalWith(other.shape)) {
        return multiply(this, other, resultDType);
      } else if (shape.compatibleWith(other.shape)) {
        return multiplyWithCompShape(this, other, resultDType);
      } else if (shape.equalWithLastDims(other.shape)) {
        return multiplyWithLastDims(
            rank > other.rank ? this : other,
            rank < other.rank ? this : other,
            resultDType);
      } else if (other.shape.size == 1) {
        return multiplyScalar(this, other.buffer[0], resultDType);
      } else {
        throw ArgumentError.value(other.shape, '',
            'Error shape: ${other.shape} is not compatible with $shape');
      }
    } else {
      throw ArgumentError(
          'Expected num or Tensor, but got ${other.runtimeType}',
          'loredart_tensor');
    }
  }

  @override
  Tensor operator /(Object other) {
    if (other is num) {
      final resultDType = dTypeAndNumDecision(dType, other.runtimeType);
      return divideScalar(this, other, resultDType);
    } else if (other is NumericTensor) {
      final resultDType = dTypeDecision(dType, other.dType);
      if (shape.equalWith(other.shape)) {
        return divide(this, other, resultDType);
      } else if (shape.compatibleWith(other.shape)) {
        return divideWithCompShape(this, other, resultDType);
      } else if (shape.equalWithLastDims(other.shape)) {
        return divideWithLastDims(rank > other.rank ? this : other,
            rank > other.rank ? other : this, resultDType);
      } else if (other.shape.size == 1) {
        return divideScalar(this, other.buffer[0], resultDType);
      } else {
        throw ArgumentError.value(other.shape, '',
            'Error shape: ${other.shape} is not compatible with $shape');
      }
    } else {
      throw ArgumentError(
          'Expected num or Tensor, but got ${other.runtimeType}',
          'loredart_tensor');
    }
  }

  @override
  Tensor operator -() {
    return negative(this, dType);
  }

  @override
  toString() {
    String tensorStr = _toStringAsValues();
    return '<Tensor(shape: $shape, values:\n [' +
        tensorStr +
        '], dType: ${dType.name})>';
  }

  @override
  String toStringShort() {
    return _toStringAsValues();
  }

  String _toStringAsValues() {
    if (rank == 1) {
      return buffer.toString();
    }
    int lastDim = shape[rank - 1];
    int numberOfSlices = shape.size ~/ lastDim;
    List<String> dimsString = [];
    if (lastDim == 1) {
      dimsString = List.generate(
          shape.size, (i) => buffer.sublist(i, (i + 1)).toString());
    } else {
      dimsString = List.generate(
          numberOfSlices,
          (i) =>
              buffer.sublist(i * lastDim, (i + 1) * lastDim).toString() +
              ((i + 1) % shape[rank - 2] == 0 ? '' : '\n '));
    }
    for (int i = rank - 2; i > 0; i -= 1) {
      int dim = shape[i];
      numberOfSlices = numberOfSlices ~/ dim;
      dimsString = List.generate(
          numberOfSlices,
          (j) =>
              dimsString
                  .sublist(j * dim, (j + 1) * dim)
                  .toString()
                  .replaceAll(RegExp('[,]'), '') +
              ((j + 1) % shape[i - 1] == 0 ? '' : '\n '));
    }
    return dimsString.reduce((s1, s2) => s1 + s2);
  }
}