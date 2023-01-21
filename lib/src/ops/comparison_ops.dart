import '/src/ops/other_ops.dart' show reshape;
import '/src/utils/dtype_utils.dart';
import '/src/tensors/tensor.dart';
import 'basic_ops.dart' show broadcastCompShapes;

enum ComparisonType {equal, notEqual, greater, greaterEqual, less, lessEqual}

bool compare(num x, num y, ComparisonType type) {
  if (type == ComparisonType.equal) {
    return x == y;
  } else if (type == ComparisonType.notEqual) {
    return x != y;
  } else if (type == ComparisonType.greater) {
    return x > y;
  } else if (type == ComparisonType.greaterEqual) {
    return x >= y;
  } else if (type == ComparisonType.less) {
    return x < y;
  } else if (type == ComparisonType.lessEqual) {
    return x <= y;
  } else {
    return false;
  }
}

Tensor compareTensors(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  if (x.dType != other.dType) {
    throw ArgumentError('Tensors to compare must be of the same DType, but received ${x.dType} and ${other.dType}');
  }

   if (other.shape.size == 1) {
    if (other.rank > x.rank) {
      x = reshape(x, [...List.filled(other.rank-x.rank, 1), ...x.shape.list]) as NumericTensor;
    }
    return compareWithScalar(x, other.buffer[0], type: type);
  } else if (x.shape.size == 1) {
    if (x.rank > other.rank) {
      x = reshape(other, [...List.filled(x.rank-other.rank, 1), ...other.shape.list]) as NumericTensor;
    }
    return compareWithScalar(other, x.buffer[0], type: type);
  } else if (x.shape.compatibleWith(other.shape)) {
    return compareWithCompShapes(x, other, type: type);
  } else if (x.shape.equalWithLastDims(other.shape)) {
    return compareWithLastDims(x, other, type: type);
  } else if (other.shape.equalWithLastDims(x.shape)) {
    return compareWithLastDims(other, x, type: type);
  } else if (x.shape.equalWith(other.shape)) {
    return compareWithEqualShapes(x, other, type: type);
  } else {
    throw ArgumentError('Tensors must be with compilable shapes, but received ${x.shape} and ${other.shape}');
  }
}

Tensor compareWithScalar(NumericTensor x, num scalar, {required ComparisonType type}) {
  List buffer = emptyBuffer(x.dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if(compare(x.buffer[i], scalar, type)) {
      buffer[i] = x.dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: x.dType);
}

Tensor compareWithEqualShapes(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  final dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  for (int i = 0; i < x.shape.size; i += 1) {
    if(compare(x.buffer[i], other.buffer[i], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
}

Tensor compareWithCompShapes(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  final dType = dTypeDecision(x.dType, other.dType);
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
    if (compare(x.buffer[tIndex], other.buffer[otherIndex], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

Tensor compareWithLastDims(NumericTensor x, NumericTensor other, {required ComparisonType type}) {
  if (other.shape.size > x.shape.size) {
    throw ArgumentError('Incorrect arguments order: expect to have x with bigger size than other');
  }
  final dType = dTypeDecision(x.dType, other.dType);
  List buffer = emptyBuffer(dType, x.shape.size);
  final int residualSize = x.shape.list.sublist(0, x.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      if (compare(x.buffer[b*matchSize + i], other.buffer[i], type)) {
        buffer[b*matchSize + i] = dType.isInt ? 1 : 1.0;
      }
    }
  }
  return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
}


NumericTensor convertToNumericTensor(dynamic object) {
  if (object is NumericTensor) {
    return object;
  }
  else if (object is num) {
    return Tensor.constant([object], shape: [1]) as NumericTensor;
  } else {
    throw ArgumentError('Expected NumericalTensor or a num, but received ${object.runtimeType}');
  }
}

Tensor equal(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.equal);
}

Tensor greater(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greater);
}

Tensor greaterEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greaterEqual);
}

Tensor less(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.less);
}

Tensor lessEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.lessEqual);
}

Tensor notEqual(Tensor x, dynamic y) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x to a NumericTensor, but received x of ${x.dType}');
  }
  NumericTensor v = convertToNumericTensor(y);
  return compareTensors(x, v, type: ComparisonType.notEqual);
}
