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

Tensor compareTensors(NumericTensor x, NumericTensor y, {required ComparisonType type, DType? dType}) {
  if (x.dType != y.dType) {
    throw ArgumentError('Tensors to compare must be of the same DType, but received ${x.dType} and ${y.dType}');
  }
  dType ??= x.dType;

  if (x.shape.size == 1 || y.shape.size == 1) {
    return compareWithScalar(x.shape.size == 1 ? y : x, x.shape.size == 1 ? x.buffer[0] : y.buffer[0], type: type, dType: dType);
  } else if (x.shape.compatibleWith(y.shape)) {
    return compareWithCompShapes(x, y, type: type, dType: dType);
  } else if (x.shape.equalWithLastDims(y.shape)) {
    return compareWithLastDims(x, y, type: type, dType: dType);
  } else if (y.shape.equalWithLastDims(x.shape)) {
    return compareWithLastDims(y, x, type: type, dType: dType);
  } else if (x.shape.equalWith(y.shape)) {
    return compareWithEqualShapes(x, y, type: type, dType: dType);
  } else {
    throw ArgumentError('Tensors must be with compilable shapes, but received ${x.shape} and ${y.shape}');
  }
}

Tensor compareWithScalar(NumericTensor t, num scalar, {required ComparisonType type, required DType dType}) {
  List buffer = emptyBuffer(dType, t.shape.size);
  for (int i = 0; i < t.shape.size; i += 1) {
    if(compare(t.buffer[i], scalar, type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}

Tensor compareWithEqualShapes(NumericTensor t, NumericTensor other, {required ComparisonType type, required DType dType}) {
  List buffer = emptyBuffer(dType, t.shape.size);
  for (int i = 0; i < t.shape.size; i += 1) {
    if(compare(t.buffer[i], other.buffer[i], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}

Tensor compareWithCompShapes(NumericTensor t, NumericTensor other, {required ComparisonType type, required DType dType}) {
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
    if (compare(t.buffer[tIndex], other.buffer[otherIndex], type)) {
      buffer[i] = dType.isInt ? 1 : 1.0;
    }
  }
  return Tensor.fromBuffer(buffer, shape, dType: dType);
}

Tensor compareWithLastDims(NumericTensor t, NumericTensor other, {required ComparisonType type, required DType dType}) {
  if (other.shape.size > t.shape.size) {
    throw ArgumentError('Incorrect arguments order: expect to have t with bigger size than other');
  }
  List buffer = emptyBuffer(dType, t.shape.size);
  final int residualSize = t.shape.list.sublist(0, t.rank-other.rank).reduce((e1, e2) => e1*e2);
  final int matchSize = other.shape.size;
  for (int b = 0; b < residualSize; b += 1) {
    for (int i = 0; i < matchSize; i += 1) {
      if (compare(t.buffer[b*matchSize + i], other.buffer[i], type)) {
        buffer[b*matchSize + i] = dType.isInt ? 1 : 1.0;
      }
    }
  }
  return Tensor.fromBuffer(buffer, t.shape.list, dType: dType);
}


NumericTensor convertToNumvericTensor(dynamic object) {
  if (object is NumericTensor) {
    return object;
  }
  else if (object is num) {
    return Tensor.constant([object], shape: [1]) as NumericTensor;
  } else {
    throw ArgumentError('Expected NumericalTensor or a num, but received ${object.runtimeType}');
  }
}

Tensor equal(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.equal, dType: dtype);
}

Tensor greater(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greater, dType: dtype);
}

Tensor greaterEqual(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.greaterEqual, dType: dtype);
}

Tensor less(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.less, dType: dtype);
}

Tensor lessEqual(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.lessEqual, dType: dtype);
}

Tensor notEqual(Tensor x, dynamic y, {DType? dtype}) {
  if (x is! NumericTensor) {
    throw ArgumentError('Expected x and y must be numeric tensors, but received x: ${x.dType}');
  }
  NumericTensor v = convertToNumvericTensor(y);
  return compareTensors(x, v, type: ComparisonType.notEqual, dType: dtype);
}
