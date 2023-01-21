import '../tensors/tensor.dart';
import '../utils/dtype_utils.dart';

Tensor negative(NumericTensor x, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = -x.buffer[i];
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Adds [other] to elements of [x].
/// 
/// Returns [Tensor] of given [dType].
Tensor addScalar(NumericTensor x, num other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] + other;
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Subtracts [other] to elements of [x].
/// 
/// Returns [Tensor] of given [dType].
Tensor subtractScalar(NumericTensor x, num other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] - other;
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Multiplies [other] to elements of [x].
/// 
/// Returns [Tensor] of given [dType].
Tensor multiplyScalar(NumericTensor x, num other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] * other;
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Divides [other] to elements of [x].
/// 
/// Returns [Tensor] of given [dType].
Tensor divideScalar(NumericTensor x, num other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] / other;
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}


/// Adds [x] and [other] element-wise.
/// 
/// Returns [Tensor] of given [dType].
Tensor add(NumericTensor x, NumericTensor other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] + other.buffer[i];
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Subtracts [x] and [other] element-wise.
/// 
/// Returns [Tensor] of given [dType].
Tensor subtract(NumericTensor x, NumericTensor other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] - other.buffer[i];
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Multiplies [x] and [other] element-wise.
/// 
/// Returns [Tensor] of given [dType].
Tensor multiply(NumericTensor x, NumericTensor other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = x.buffer[i] * other.buffer[i];
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}

/// Divides [x] and [other] element-wise.
/// 
/// Returns [Tensor] of given [dType].
Tensor divide(NumericTensor x, NumericTensor other, DType dType) {
  List resultBuffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    resultBuffer[i] = dType.isInt ? (x.buffer[i] / other.buffer[i] as num).toInt() : x.buffer[i] / other.buffer[i];
  }
  return Tensor.fromBuffer(resultBuffer, x.shape.list, dType: dType);
}