// ignore_for_file: overridden_fields

import 'dart:typed_data';

import 'tensor.dart';
import 'tensor_shape.dart';
import '../ops/basic_ops.dart';
import '../utils/diagonal_utils.dart';

/// A Tensor with numeric elements and [dType].
/// 
/// [NumericTensor] is a concrete implementation of [Tensor] that stores num values in a TypedData List [buffer].
abstract class NumericTensor<L extends List> implements Tensor {

  /// A storage list of tensors values.
  /// 
  /// Usually is one of the [TypedData] lists.
  late final L buffer;

  @override
  late final TensorShape shape;
  
  @override
  late final DType dType; 

  @override
  int get rank => shape.rank;

  /// Returns [this] + [other] element-wise.
  /// 
  /// [other] might be a Dart [num] compatible with [dType],
  /// or a [NumericTensor] with broadcastable shape and same [dType] as [this].
  /// 
  /// If [other] neither [num] nor [NumericTensor] will throw an ArgumentError.
  @override
  Tensor operator +(Object other) {
    if (other is num) {
      return addScalar(this, other);
    } else if (other is NumericTensor) {
      return numericOperation(this, other, ArithmeticOperation.add);
    } else {
      throw ArgumentError('Expected num or NumericTensor (of the same DType) as other, but received ${other.runtimeType}', 'other');
    }
  }

  /// Returns [this] - [other] element-wise.
  /// 
  /// [other] might be a Dart [num] compatible with [dType],
  /// or a [NumericTensor] with broadcastable shape and same [dType] as [this].
  /// 
  /// If [other] neither [num] nor [NumericTensor] will throw an ArgumentError.
  @override
  Tensor operator -(Object other) {
    if (other is num) {
      return subtractScalar(this, other);
    } else if (other is NumericTensor) {
      return numericOperation(this, other, ArithmeticOperation.subtract);
    } else {
      throw ArgumentError('Expected num or NumericTensor (of the same DType) as other, but received ${other.runtimeType}', 'other');
    }
  }

  /// Returns [this] * [other] element-wise.
  /// 
  /// [other] might be a Dart [num] compatible with [dType],
  /// or a [NumericTensor] with broadcastable shape and same [dType] as [this].
  /// 
  /// If [other] neither [num] nor [NumericTensor] will throw an ArgumentError.
  @override
  Tensor operator *(Object other) {
    if (other is num) {
      return multiplyScalar(this, other);
    } else if (other is NumericTensor) {
      return numericOperation(this, other, ArithmeticOperation.multiply);
    } else {
      throw ArgumentError('Expected num or NumericTensor (of the same DType) as other, but received ${other.runtimeType}', 'other');
    }
  }

  /// Returns [this] / [other] element-wise.
  /// 
  /// [other] might be a Dart [num] compatible with [dType],
  /// or a [NumericTensor] with broadcastable shape and same [dType] as [this].
  /// 
  /// If [other] neither [num] nor [NumericTensor] will throw an ArgumentError.
  @override
  Tensor operator /(Object other) {
    if (other is num) {
      return divideScalar(this, other);
    } else if (other is NumericTensor) {
      return numericOperation(this, other, ArithmeticOperation.divide);
    } else {
      throw ArgumentError('Expected num or NumericTensor (of the same DType) as other, but received ${other.runtimeType}', 'other');
    }
  }

  @override
  Tensor operator -() {
    return negative(this);
  }

  @override
  toString() {
    String tensorStr = _toStringAsValues();
    return 'Tensor(shape: $shape, values:\n [' +
        tensorStr +
        '], dType: ${dType.name})';
  }

  @override
  String toStringShort() {
    return _toStringAsValues();
  }

  /// String representation of the values from [buffer], which consider the [shape].
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

/// [NumericTensor] with float32 values stored in [Float32List].
class Float32NumericTensor extends NumericTensor<Float32List> {
  @override
  final dType = DType.float32;

  Float32NumericTensor.zeros(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size);
  }

  Float32NumericTensor.fromTypedDataList(Float32List buffer, List<int> shape) {
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Float32NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {double value = 1.0}) {
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Float32List;
  }

  Float32NumericTensor.fill(List<int> shape, double value) {
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Float32NumericTensor.ones(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size)..fillRange(0, this.shape.size, 1.0);
  }
}

/// [NumericTensor] with float64 values stored in [Float64List].
class Float64NumericTensor extends NumericTensor<Float64List> {
  @override
  final dType = DType.float64;

  Float64NumericTensor.zeros(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size);
  }

  Float64NumericTensor.fromTypedDataList(Float64List buffer, List<int> shape) {
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Float64NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {double value = 1.0}) {
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = 
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Float64List;
  }

  Float64NumericTensor.fill(List<int> shape, double value) {
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Float64NumericTensor.ones(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size)..fillRange(0, this.shape.size, 1.0);
  }
}

/// [NumericTensor] with int32 values stored in [Int32List].
class Int32NumericTensor extends NumericTensor<Int32List> {
  @override
  final dType = DType.int32;

  Int32NumericTensor.zeros(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size);
  }

  Int32NumericTensor.fromTypedDataList(Int32List buffer, List<int> shape) {
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Int32NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {int value = 1}) {
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Int32List;
  }

  Int32NumericTensor.fill(List<int> shape, int value) {
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Int32NumericTensor.ones(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size)..fillRange(0, this.shape.size, 1);
  }
}

/// [NumericTensor] with int64 values stored in [Int64List].
class Int64NumericTensor extends NumericTensor<Int64List> {
  @override
  final dType = DType.int64;

  Int64NumericTensor.zeros(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size);
  }

  Int64NumericTensor.fromTypedDataList(Int64List buffer, List<int> shape) {
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Int64NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {int value = 1}) {
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Int64List;
  }

  Int64NumericTensor.fill(List<int> shape, int value) {
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Int64NumericTensor.ones(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size)..fillRange(0, this.shape.size, 1);
  }
}


/// [NumericTensor] with uint8 values stored in [Uint8List].
class Uint8NumericTensor extends NumericTensor<Uint8List> {
  @override
  final dType = DType.uint8;

  Uint8NumericTensor.zeros(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Uint8List(this.shape.size);
  }

  Uint8NumericTensor.fromTypedDataList(Uint8List buffer, List<int> shape) {
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Uint8NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {int value = 1}) {
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Uint8List;
  }

  Uint8NumericTensor.fill(List<int> shape, int value) {
    this.shape = TensorShape(shape);
    buffer = Uint8List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Uint8NumericTensor.ones(List<int> shape) {
    this.shape = TensorShape(shape);
    buffer = Uint8List(this.shape.size)..fillRange(0, this.shape.size, 1);
  }
}