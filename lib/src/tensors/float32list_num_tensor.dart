import 'dart:typed_data';

import '../utils/diagonal_utils.dart';
import 'tensor.dart';
import 'tensor_shape.dart';


class Float32NumericTensor extends NumericTensor<Float32List> {

  Float32NumericTensor.zeros(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size);
  }

  Float32NumericTensor.fromBuffer(Float32List buffer, List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Float32NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {double value = 1.0}) {
    initDType();
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Float32List;
  }

  Float32NumericTensor.fill(List<int> shape, double value) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Float32NumericTensor.ones(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float32List(this.shape.size)..fillRange(0, this.shape.size, 1.0);
  }

  initDType() => dType = DType.float32;
}