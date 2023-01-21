import 'dart:typed_data';
import '../tensors/tensor.dart';


DType dTypeDecision(DType dType1, DType dType2) {
  if (dType1 == dType2) {
    return dType1;
  } else {
    throw ArgumentError('Tensors must be of the same DType, but received $dType1 and $dType2');
  }
}

DType dTypeAndNumDecision(DType dType, Type type, [bool tensorInitialization = false]) {
  if ((dType == DType.float32) && type == double) {
    return DType.float32;
  } else if (dType == DType.float64 && type == double) {
    return DType.float64;
  } else if (dType == DType.int32 && type == int) {
    return DType.int32;
  } else if (dType == DType.int64 && type == int) {
    return DType.int64;
  }  else if (dType == DType.float32 && type == int) {
    return DType.float32;
  } else if (dType == DType.float64 && type == int) {
    return DType.float64;
  } else if ((dType == DType.int32 || dType == DType.int64) && type == double && tensorInitialization) {
    return dType;
  } else {
    throw UnsupportedError('$dType and $type are not compatible data types.');
  }
}

List emptyBuffer(DType dType, int length) {
  if (dType == DType.float32) {
    return Float32List(length);
  } else if (dType == DType.float64) {
    return Float64List(length);
  } else if (dType == DType.int32) {
    return Int32List(length);
  } else if (dType == DType.int64) {
    return Int64List(length);
  }
  else {
    throw UnsupportedError('$dType is not supported for empty buffer');
  }
}

