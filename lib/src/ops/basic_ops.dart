import 'dart:math';

import '/src/ops/other_ops.dart' show reshape;
import '../tensors/tensor.dart';
import '../utils/dtype_utils.dart';
import '/src/tensors/tensor_shape.dart';

enum OperationType {add, subtract, multiply, divide}

/// Computes resulting shape of a Tensor from broadcastable [TensorShape]s [ts1] and [ts2]
List<int> broadcastCompShapes(TensorShape ts1, TensorShape ts2) {
  List<int> shapeBase = [];
  for (int i = 0; i < ts1.rank; i += 1) {
    shapeBase.add(max(ts1[i], ts2[i]));
  }
  return shapeBase;
}

/// Computes element-wise negative of the elements of [x]
/// 
/// Returns [Tensor] of the same [DType] as [x]
Tensor negative(NumericTensor x) {
  List buffer = emptyBuffer(x.dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    buffer[i] = -x.buffer[i];
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: x.dType);
}

Tensor opWithScalar(NumericTensor x, num other, OperationType operationType) {
  DType dType = dTypeAndNumDecision(x.dType, other.runtimeType);
  List buffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if (operationType == OperationType.add) {
      buffer[i] = x.buffer[i]+other;
    }
    else if (operationType == OperationType.subtract) {
      buffer[i] = x.buffer[i]-other;
    }
    else if (operationType == OperationType.multiply) {
      buffer[i] = x.buffer[i]*other;
    }
    else {
      buffer[i] = x.buffer[i]/other;
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
}

/// Adds [other] to elements of [x].
/// 
/// Returns [Tensor] of derived [DType] from [x.dType] and [other.runtimeType].
Tensor addScalar(NumericTensor x, num other) => opWithScalar(x, other, OperationType.add);

/// Subtracts [other] to elements of [x].
/// 
/// Returns [Tensor] of derived [DType] from [x.dType] and [other.runtimeType].
Tensor subtractScalar(NumericTensor x, num other) => opWithScalar(x, other, OperationType.subtract);

/// Multiplies [other] to elements of [x].
/// 
/// Returns [Tensor] of derived [DType] from [x.dType] and [other.runtimeType].
Tensor multiplyScalar(NumericTensor x, num other) => opWithScalar(x, other, OperationType.multiply);

/// Divides [other] to elements of [x].
/// 
/// Returns [Tensor] of derived [DType] from [x.dType] and [other.runtimeType].
Tensor divideScalar(NumericTensor x, num other) => opWithScalar(x, other, OperationType.divide);


/// Adds [x] and [other] element-wise.
/// 
/// Returns [Tensor] of given [dType].
Tensor opWithEqualShapes(NumericTensor x, NumericTensor other, OperationType operationType) {
  DType dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if (operationType == OperationType.add) {
      buffer[i] = x.buffer[i]+other.buffer[i];
    }
    else if (operationType == OperationType.subtract) {
      buffer[i] = x.buffer[i]-other.buffer[i];
    }
    else if (operationType == OperationType.multiply) {
      buffer[i] = x.buffer[i]*other.buffer[i];
    }
    else {
      buffer[i] = x.buffer[i]/other.buffer[i];
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
}


/// Computes "element-wise" [operation] of Tensors [x] and [other] with compatible shapes.
Tensor opWithCompShape(NumericTensor x, NumericTensor other, OperationType operationType) {
  DType dType = dTypeDecision(x.dType, other.dType);
  final List<int> shape = broadcastCompShapes(x.shape, other.shape);
  final int length = shape.reduce((a,b) => a*b);

  final List<int> cumProdT = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : (x.shape[i] == 1 ? 0 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
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
      tIndex += x.shape[k] == 1 ? 0 : cumProdT[k] * currentIndices[k];
      otherIndex += other.shape[k] == 1 ? 0 : cumProdOther[k] * currentIndices[k];
    }
    if (operationType == OperationType.add) {
      buffer[i] = x.buffer[tIndex]+other.buffer[otherIndex];
    }
    else if (operationType == OperationType.subtract) {
      buffer[i] = x.buffer[tIndex]-other.buffer[otherIndex];
    }
    else if (operationType == OperationType.multiply) {
      buffer[i] = x.buffer[tIndex]*other.buffer[otherIndex];
    }
    else {
      buffer[i] = x.buffer[tIndex]/other.buffer[otherIndex];
    }
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}


/// Computes "element-wise" [operation] of Tensors [x] and [other] with equal last k dims of shape.
/// 
/// [x] must be a [Tensor] of higher rank, i.e., [x.rank] > [other.rank].
Tensor opWithLastDims(NumericTensor x, NumericTensor other, OperationType operationType) {
  if (other.rank > x.rank) {
    throw ArgumentError('Incorrect arguments order: expect to have x with higher rank than other, but received x.rank: ${x.rank} and other.rank: ${other.rank}');
  }
  DType dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  final int residualSize = x.shape.list.sublist(0, x.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;

  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      if (operationType == OperationType.add) {
        buffer[b*matchSize + i] = x.buffer[b*matchSize+i]+other.buffer[i];
      }
      else if (operationType == OperationType.subtract) {
        buffer[b*matchSize + i] = x.buffer[b*matchSize+i]-other.buffer[i];
      }
      else if (operationType == OperationType.multiply) {
        buffer[b*matchSize + i] = x.buffer[b*matchSize+i]*other.buffer[i];
      }
      else {
        buffer[b*matchSize + i] = x.buffer[b*matchSize+i]/other.buffer[i];
      }
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
}

/// Returns element-wise [operation] of [x] and [other].
/// 
/// [x] and [other] must be of the same [DType]. 
/// 
/// Tensors might be with different, but should be with broadcastable [TensorShape]s. 
Tensor numericOperation(NumericTensor x, NumericTensor other, OperationType operationType) {
  if (x.dType != other.dType) {
    throw ArgumentError('Tensors must be of the same DType, but received ${x.dType} and ${other.dType}');
  }
  if (x.shape.equalWith(other.shape)) {
      return opWithEqualShapes(x, other, operationType);
    } else if (x.shape.compatibleWith(other.shape)) {
      return opWithCompShape(x, other, operationType);
    } else if (x.shape.equalWithLastDims(other.shape)) {
      return opWithLastDims(x.rank > other.rank ? x : other, x.rank > other.rank ? other : x, operationType);
    } else if (other.shape.size == 1) {
      if (other.rank > x.rank) {
        x = reshape(x, [...List.filled(other.rank-x.rank, 1), ...x.shape.list]) as NumericTensor;
      }
      return opWithScalar(x, other.buffer[0], operationType);
    } else if (x.shape.size == 1) {
      if (x.rank > other.rank) {
        x = reshape(other, [...List.filled(x.rank-other.rank, 1), ...other.shape.list]) as NumericTensor;
      }
      return opWithScalar(other, x.buffer[0], operationType);
    }  else {
      throw ArgumentError('Tensor shape ${x.shape} is not compatible with ${other.shape}', 'other');
    }
}