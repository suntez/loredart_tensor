import 'dart:math';

import '../tensors/tensor.dart';
import '../tensors/tensor_shape.dart';
import '../utils/dtype_utils.dart';

/// Computes resulting shape of a Tensor from broadcastable [TensorShape]s [ts1] and [ts2]
List<int> broadcastCompShapes(TensorShape ts1, TensorShape ts2) {
  List<int> shapeBase = [];
  for (int i = 0; i < ts1.rank; i += 1) {
    shapeBase.add(max(ts1[i], ts2[i]));
  }
  return shapeBase;
}

/// Computes "element-wise" addition of Tensors [t] and [other] with compatible shapes.
Tensor addWithCompShape(NumericTensor t, NumericTensor other, DType dType) {
  final List<int> shape = broadcastCompShapes(t.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (t.shape[i] == 1 ? 0 : t.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
  final List<int> cumProdOther = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (other.shape[i] == 1 ? 0 : other.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));

  List<int> currentIndices = List<int>.filled(shape.length, 0);
  List buffer = emptyBuffer(dType, length);
  int tIndex = 0;
  int otherIndex = 0;
  for (int i = 0; i < length; i += 1) {

    tIndex = 0;
    otherIndex = 0;

    int index = i;
      for (int j = shape.length - 1; j >= 0; j -= 1) {
        currentIndices[j] = index % shape[j];
        index = index ~/ shape[j];
    }
    for (int k = 0; k < shape.length; k += 1) {
      tIndex += t.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }

    buffer[i] = t.buffer[tIndex] + other.buffer[otherIndex];
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

/// Computes "element-wise" subtraction of Tensors [t] and [other] with compatible shapes.
Tensor subtractWithCompShape(NumericTensor t, NumericTensor other, DType dType) {
  final List<int> shape = broadcastCompShapes(t.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (t.shape[i] == 1 ? 0 : t.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
  final List<int> cumProdOther = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (other.shape[i] == 1 ? 0 : other.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));

  List<int> currentIndices = List<int>.filled(shape.length, 0);
  List buffer = emptyBuffer(dType, length);
  int tIndex = 0;
  int otherIndex = 0;
  for (int i = 0; i < length; i += 1) {

    tIndex = 0;
    otherIndex = 0;

    int index = i;
      for (int j = shape.length - 1; j >= 0; j -= 1) {
        currentIndices[j] = index % shape[j];
        index = index ~/ shape[j];
    }
    for (int k = 0; k < shape.length; k += 1) {
      tIndex += t.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }

    buffer[i] = t.buffer[tIndex] - other.buffer[otherIndex];
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

/// Computes "element-wise" multiplication of Tensors [t] and [other] with compatible shapes.
Tensor multiplyWithCompShape(NumericTensor t, NumericTensor other, DType dType) {
  final List<int> shape = broadcastCompShapes(t.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (t.shape[i] == 1 ? 0 : t.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
  final List<int> cumProdOther = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (other.shape[i] == 1 ? 0 : other.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));

  List<int> currentIndices = List<int>.filled(shape.length, 0);
  List buffer = emptyBuffer(dType, length);
  int tIndex = 0;
  int otherIndex = 0;
  for (int i = 0; i < length; i += 1) {

    tIndex = 0;
    otherIndex = 0;

    int index = i;
      for (int j = shape.length - 1; j >= 0; j -= 1) {
        currentIndices[j] = index % shape[j];
        index = index ~/ shape[j];
    }
    for (int k = 0; k < shape.length; k += 1) {
      tIndex += t.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }

    buffer[i] = t.buffer[tIndex] * other.buffer[otherIndex];
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

/// Computes "element-wise" division of Tensors [t] and [other] with compatible shapes.
Tensor divideWithCompShape(NumericTensor t, NumericTensor other, DType dType) {
  final List<int> shape = broadcastCompShapes(t.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (t.shape[i] == 1 ? 0 : t.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
  final List<int> cumProdOther = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (other.shape[i] == 1 ? 0 : other.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));

  List<int> currentIndices = List<int>.filled(shape.length, 0);
  List buffer = emptyBuffer(dType, length);
  // buffer[0] = t.buffer[0] / other.buffer[0];
  int tIndex = 0;
  int otherIndex = 0;
  for (int i = 0; i < length; i += 1) {

    tIndex = 0;
    otherIndex = 0;

    int index = i;
    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }
    for (int k = 0; k < shape.length; k += 1) {
      tIndex += t.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }

    buffer[i] = dType.isInt ? (t.buffer[tIndex] / other.buffer[otherIndex] as num).toInt() : t.buffer[tIndex] / other.buffer[otherIndex];
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

/// Computes "element-wise" addition of Tensors [t] and [other] with equal k dims of shape.
/// 
/// [t] must be a [Tensor] of bigger rank, i.e., [t.rank] > [other.rank].
Tensor addWithLastDims(NumericTensor t, NumericTensor other, DType dType) {
  if (other.shape.size > t.shape.size) {
    throw ArgumentError('Incorrect argument order: expect to have t with bigger size than other');
  }
  List buffer = emptyBuffer(dType, t.shape.size);
  final int residualSize = t.shape.list.sublist(0, t.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      buffer[b*matchSize + i] = t.buffer[b*matchSize + i] + other.buffer[i];
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}

/// Computes "element-wise" subtraction  of Tensors [t] and [other] with equal k dims of shape.
/// 
/// [t] must be a [Tensor] of bigger rank, i.e., [t.rank] > [other.rank].
Tensor subtractWithLastDims(NumericTensor t, NumericTensor other, DType dType) {
  if (other.shape.size > t.shape.size) {
    throw ArgumentError('Incorrect argument order: expect to have t with bigger size than other');
  }
  List buffer = emptyBuffer(dType, t.shape.size);
  print(t.shape.list.sublist(0, t.rank - other.rank));
  final int residualSize = t.shape.list.sublist(0, t.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      buffer[b*matchSize + i] = t.buffer[b*matchSize + i] - other.buffer[i];
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}

/// Computes "element-wise" multiplication of Tensors [t] and [other] with equal k dims of shape.
/// 
/// [t] must be a [Tensor] of bigger rank, i.e., [t.rank] > [other.rank].
Tensor multiplyWithLastDims(NumericTensor t, NumericTensor other, DType dType) {
  if (other.shape.size > t.shape.size) {
    throw ArgumentError('Incorrect argument order: expect to have t with bigger size than other');
  }
  List buffer = emptyBuffer(dType, t.shape.size);
  final int residualSize = t.shape.list.sublist(0, t.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;

  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      buffer[b*matchSize + i] = t.buffer[b*matchSize + i] * other.buffer[i];
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}

/// Computes "element-wise" division of Tensors [t] and [other] with equal k last dims of shape.
/// 
/// [t] must be a [Tensor] of bigger rank, i.e., [t.rank] > [other.rank].
Tensor divideWithLastDims(NumericTensor t, NumericTensor other, DType dType) {
  if (other.shape.size > t.shape.size) {
    throw ArgumentError('Incorrect argument order: expect to have t with bigger size than other');
  }
  List buffer = emptyBuffer(dType, t.shape.size);
  final int residualSize = t.shape.list.sublist(0, t.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      buffer[b*matchSize + i] = dType.isInt ? (t.buffer[b*matchSize + i] / other.buffer[i] as num).toInt() : t.buffer[b*matchSize + i] / other.buffer[i];
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}