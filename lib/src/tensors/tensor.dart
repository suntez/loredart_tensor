import 'dart:typed_data';

import 'num_tensor.dart';
import 'tensor_shape.dart';

import '/src/utils/diagonal_utils.dart';
import '/src/utils/shape_utils.dart';



/// The data types of the elements in a [Tensor].
/// 
/// `bool` and `string` data types aren't supported yet.
enum DType {
  float32,
  float64,
  uint8,
  int32,
  int64,
  string,
  bool
}

/// The types of the [DType] data types.
extension DTypeTypes on DType {
  /// True if [this] is integer-based type
  bool get isInt => this == DType.uint8 || this == DType.int32 || this == DType.int64;

  /// True if [this] is double-based type
  bool get isDouble => this == DType.float32 || this == DType.float64;

  /// True if [this] is num-based type
  bool get isNumeric => this != DType.bool && this != DType.string;
}

/// Default correspondence of Dart [Type]s and tensors' [DType]. 
const Map<Type, DType> defaultTypesToDTypes = {
  int: DType.int32,
  double: DType.float32,
  // bool: DType.bool,
  // String: DType.string
};

/// The representation of a multidimensional array of elements of a single type.
/// 
/// Each [Tensor] is described by two main properties:
/// - [shape]
/// - single [dType]
/// 
/// [Tensor]s are considered immutable and any operation with them will produce the new instance of [Tensor] class.
/// 
/// See concrete [Tensor] implementations for more details:
/// - [NumericTensor] for numeric [DTypes]
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
  /// [values] may be list of [num] or have a multidimensional (nested) structure. Non-numerical lists aren't supported yet.
  /// 
  /// If [shape] is specified, tries to reshape elements to match it, otherwise inherits shape from [values].
  /// If number of elements of [values] won't be equal to [shape.size] throws an [ArgumentError].
  /// 
  /// If [dType] is specified, casts elements of [value] to meet the type, otherwise will inherit [dType] according to `defaultTypesToDTypes`.
  /// 
  /// If [values] contains non-numerical elements, elements of different type, or nested lists have non-equal length - will throw an [ArgumentError].
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
  /// Tensor t = Tensor.constant([1,2,3,4,5,6], shape: [3,2]); // from vector to any shape
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
        return Float32NumericTensor.fromTypedDataList(
            Float32List.fromList(List.generate(flatten.length, (i) => flatten[i].toDouble(), growable: false)), shape);
      } else if (dType == DType.float64) {
        return Float64NumericTensor.fromTypedDataList(
            Float64List.fromList(List.generate(flatten.length, (i) => flatten[i].toDouble(), growable: false)), shape);
      } else if (dType == DType.uint8) {
        return Uint8NumericTensor.fromTypedDataList(
            Uint8List.fromList(List.generate(flatten.length, (i) => flatten[i].toInt(), growable: false)), shape);
      } else if (dType == DType.int32) {
        return Int32NumericTensor.fromTypedDataList(
            Int32List.fromList(List.generate(flatten.length, (i) => flatten[i].toInt(), growable: false)), shape);
      } else if (dType == DType.int64) {
        return Int64NumericTensor.fromTypedDataList(
            Int64List.fromList(List.generate(flatten.length, (i) => flatten[i].toInt(), growable: false)), shape);
      } else {
        throw UnsupportedError('DType $dType is not supported for Tensor.constant factory');
      }
    } else {
      throw ArgumentError('Cannot construct Tensor with ${flatten.length} elements of the shape $shape', 'shape');
    }
  }

  /// Creates a [Tensor] of the [dType] with [shape] and all elements set to zero.
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
    } else if (dType == DType.uint8) {
      return Uint8NumericTensor.zeros(shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.zeros(shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.zeros(shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.zeros factory', 'dType');
    }
  }

  /// Creates a [Tensor] of the [dType] with [shape] and elements from TypedData [buffer].
  /// 
  /// Assumes that [buffer] runtimeType is consistent with [dType] and [shape.size];
  /// 
  /// Throws an [ArgumentError] if [dType] is non-numeric.
  factory Tensor.fromTypedDataList(List buffer, List<int> shape,
      {required DType dType}) {
    if (buffer.length != shape.reduce((e1,e2)=>e1*e2)) {
      throw ArgumentError('Cannot allocate ${buffer.length} elements into shape $shape');
    }
    if (dType == DType.float32) {
      return Float32NumericTensor.fromTypedDataList(buffer as Float32List, shape);
    } else if (dType == DType.float64) {
      return Float64NumericTensor.fromTypedDataList(buffer as Float64List, shape);
    } else if (dType == DType.uint8) {
      return Uint8NumericTensor.fromTypedDataList(buffer as Uint8List, shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.fromTypedDataList(buffer as Int32List, shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.fromTypedDataList(buffer as Int64List, shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.fromTypedDataList', 'dType');
    }
  }

  /// Creates an identity matrix (or batch of matrices) of the [dType].
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
    } else if (dType == DType.uint8) {
      return Uint8NumericTensor.eye(
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

  /// Creates a [Tensor] of the [dType] with [shape] and all elements set to [value].
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
    } else if (dType == DType.uint8) {
      return Uint8NumericTensor.fill(shape, (value as num).toInt());
    } else if (dType == DType.int32) {
      return Int32NumericTensor.fill(shape, (value as num).toInt());
    } else if (dType == DType.int64) {
      return Int64NumericTensor.fill(shape, (value as num).toInt());
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.fill factory', 'dType');
    }
  }

  /// Creates a [Tensor] of the [dType] with [shape] and all elements set to one.
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
    } else if (dType == DType.uint8) {
      return Uint8NumericTensor.ones(shape);
    } else if (dType == DType.int32) {
      return Int32NumericTensor.ones(shape);
    } else if (dType == DType.int64) {
      return Int64NumericTensor.ones(shape);
    } else {
      throw ArgumentError(
          'DType $dType is not supported for Tensor.ones factory', 'dType');
    }
  }

  /// Creates a diagonal [Tensor] (or a batch of them) of the [dType] with given [diagonal] elements and everything else padded with zeros.
  /// 
  /// The [diagonal] might be a list of nums for creating a matrix or a nested list for making the batched 3D [Tensor].
  /// 
  /// By default, the diagonal elements will be located along the main diagonal, but it can be changed with the [offset].
  /// Specifying the [offset] will increase the size of the dimensions by its absolute value.
  /// 
  /// By default it returns squared matrix(-ces).
  /// If [numRows] and/or [numCols] are specified, it will pad extra rows and/or columns with zeros.
  /// If [numRows] or [numCols] is smaller than the number of diagonal elements will throw [ArgumentError].
  ///
  ///
  ///Example:
  ///```dart
  /// final diagonal = [1,2];
  /// 
  /// var x = Tensor.diag(diagonal);
  /// print(x); // <Tensor(shape: [2, 2], values: [[1.0, 0.0], [0.0, 2.0]], dType: float32)>
  /// 
  /// var y = print(Tensor.diag(diagonal, offset: -1));
  /// print(y);
  /// //<Tensor(shape: [3, 3], values:
  /// // [[0.0, 0.0, 0.0]
  /// //  [1.0, 0.0, 0.0]
  /// //  [0.0, 2.0, 0.0]], dType: float32)>
  /// 
  /// var t = Tensor.diag(diagonal, numCols: 3, numRows: 4);
  /// print(t.shape); // [4, 3]
  /// 
  /// final batchDiag = [[1,2,3], [3,4,5]];
  /// var v = Tensor.diag(batchDiag);
  /// print(v.shape); // [2, 3, 3]
  /// 
  /// var v2 = Tensor.diag(batchDiag, offset: 1, numRows: 5);
  /// print(v2.shape); // [2, 5, 4]
  ///```
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

  /// Returns [this] + [other] element-wise.
  /// 
  /// See concrete tensor implementations (like [NumericTensor]) for more info.
  Tensor operator +(Object other);

  /// Returns [this] - [other] element-wise.
  /// 
  /// See concrete tensor implementations (like [NumericTensor]) for more info.
  Tensor operator -(Object other);

  /// Returns [this] * [other] element-wise.
  /// 
  /// See concrete tensor implementations (like [NumericTensor]) for more info.
  Tensor operator *(Object other);

  /// Returns [this] / [other] element-wise.
  /// 
  /// See concrete tensor implementations (like [NumericTensor]) for more info.
  Tensor operator /(Object other);

  /// Returns element-wise negative of [this].
  /// 
  /// See concrete tensor implementations (like [NumericTensor]) for more info.
  Tensor operator -();

  @override
  String toString() => toStringShort();

  String toStringShort() {
    return '<Tensor(shape: $shape, dtype: $dType)>';
  }
}