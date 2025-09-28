import 'dart:typed_data';
import '../tensors/tensor.dart';

/// Checks if [dType1] == [dTypes2] and returns corresponding [DType],
/// otherwise will throw an ArgumentError.
DType dTypeDecision(DType dType1, DType dType2) {
  if (dType1 == dType2) {
    return dType1;
  } else {
    throw ArgumentError('Tensors must be of the same DType, but received $dType1 and $dType2');
  }
}

/// Returns the resulting [DType] of the operation on the Tensor of [dType] and num of [type].
///
/// If the combination of [dType] and [type] is incompatible, throws an ArgumentError.
DType dTypeAndNumDecision(DType dType, Type type, [bool tensorInitialization = false]) {
  if ((dType == DType.float32) && type == double) {
    return DType.float32;
  } else if (dType == DType.float64 && type == double) {
    return DType.float64;
  } else if (dType == DType.uint8 && type == int) {
    return DType.int32;
  } else if (dType == DType.int32 && type == int) {
    return DType.int32;
  } else if (dType == DType.int64 && type == int) {
    return DType.int64;
  } else if (dType == DType.float32 && type == int) {
    return DType.float32;
  } else if (dType == DType.float64 && type == int) {
    return DType.float64;
  } else if ((dType == DType.int32 || dType == DType.int64) && type == double && tensorInitialization) {
    return dType;
  } else {
    throw ArgumentError('$dType and $type are not compatible data types');
  }
}

/// Returns an empty buffer list of the given [length] according to the [dType].
///
/// If [dType] is a numeric one, it will return one of the TypedData Lists,
/// but if [dType] is not supported yet - it will throw an ArgumentError.
///
/// [length] must be a positive integer, otherwise throws an ArgumentError.
List emptyBuffer(DType dType, int length) {
  if (length <= 0) {
    throw ArgumentError('Length of a buffer list must be a positive integer, but received $length');
  }
  if (dType == DType.float32) {
    return Float32List(length);
  } else if (dType == DType.float64) {
    return Float64List(length);
  } else if (dType == DType.uint8) {
    return Uint8List(length);
  } else if (dType == DType.int32) {
    return Int32List(length);
  } else if (dType == DType.int64) {
    return Int64List(length);
  } else {
    throw ArgumentError('$dType is not supported for empty buffer', 'dType');
  }
}

/// Convert [byteBuffer] into a TypedData list according to the [dType].
List convertBufferToTypedDataList(ByteBuffer byteBuffer, DType dType) {
  if (dType == DType.float32) {
    return byteBuffer.asFloat32List();
  } else if (dType == DType.float64) {
    return byteBuffer.asFloat64List();
  } else if (dType == DType.uint8) {
    return byteBuffer.asUint8List();
  } else if (dType == DType.int32) {
    return byteBuffer.asInt32List();
  } else if (dType == DType.int64) {
    return byteBuffer.asInt64List();
  } else {
    throw UnsupportedError('DType $dType is not supported');
  }
}
