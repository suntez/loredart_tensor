import 'dart:typed_data';

import '../utils/diagonal_utils.dart';
import 'tensor.dart';
import 'tensor_shape.dart';

class Int32NumericTensor extends NumericTensor<Int32List> {

  Int32NumericTensor.zeros(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size);
  }

  Int32NumericTensor.fromBuffer(Int32List buffer, List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Int32NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {int value = 1}) {
    initDType();
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Int32List;
  }

  Int32NumericTensor.fill(List<int> shape, int value) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Int32NumericTensor.ones(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int32List(this.shape.size)..fillRange(0, this.shape.size, 1);
  }

  initDType() => dType = DType.int32;
}