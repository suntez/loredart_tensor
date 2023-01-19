import 'dart:typed_data';

import '../utils/diagonal_utils.dart';
import 'tensor.dart';
import 'tensor_shape.dart';

class Float64NumericTensor extends NumericTensor<Float64List> {

  Float64NumericTensor.zeros(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size);
  }

  Float64NumericTensor.fromBuffer(Float64List buffer, List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    this.buffer = buffer;
  }

  Float64NumericTensor.eye(int numRows, int numCols, List<int> batchShape, {double value = 1.0}) {
    initDType();
    List<int> shape = batchShape + [numRows, numCols];
    this.shape = TensorShape(shape);
    buffer = 
    buffer = eyeBuffer(numRows, numCols, batchShape, dType) as Float64List;
  }

  Float64NumericTensor.fill(List<int> shape, double value) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size)..fillRange(0, this.shape.size, value);
  }

  Float64NumericTensor.ones(List<int> shape) {
    initDType();
    this.shape = TensorShape(shape);
    buffer = Float64List(this.shape.size)..fillRange(0, this.shape.size, 1.0);
  }

  initDType() => dType = DType.float64;

}