import 'dart:typed_data';

import '../utils/diagonal_utils.dart';
import 'tensor.dart';
import 'tensor_shape.dart';

class Int64NumericTensor extends NumericTensor<Int64List> {

  Int64NumericTensor.zeros(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size);
  }

  Int64NumericTensor.fromBuffer(Int64List buffer, List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Int64NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {int value = 1}) {
    initDType();
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Int64List;
  }

  Int64NumericTensor.fill(List<int> shape, int value) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Int64NumericTensor.ones(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Int64List(this.shape.size)..fillRange(0, this.shape.size, 1);
  }

  initDType() => dType = DType.int64;
}