import 'dart:math' show sqrt;
import '../tensors/tensor.dart';

/// Computes the `max` of a slice from [buffer] according to [reduce] and [axis].
num reduceMaxSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num? maxValue;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    if (maxValue == null || maxValue < buffer[indexForTensor]) {
      maxValue = buffer[indexForTensor];
    }
  }
  return maxValue!;
}

/// Computes the `min` of a slice from [buffer] according to [reduce] and [axis].
num reduceMinSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num? minValue;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    if (minValue == null || minValue > buffer[indexForTensor]) {
      minValue = buffer[indexForTensor];
    }
  }
  return minValue!;
}

/// Computes the `local arg max` of a slice from [buffer] according to [reduce] and [axis].
int reduceLocalArgMaxSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num? maxValue;
  int maxValueIndex = 0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    if (maxValue == null || maxValue < buffer[indexForTensor]) {
      maxValue = buffer[indexForTensor];
      maxValueIndex = i;
    }
  }
  return maxValueIndex;
}

/// Computes the `sum` of a slice from [buffer] according to [reduce] and [axis].
num reduceSumSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num sumValue = dType.isInt ? 0 : 0.0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor +=
          cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]); // for reduce[k] == 1 cumProd is 0
    }

    sumValue += buffer[indexForTensor];
  }
  return sumValue;
}

/// Computes the `mean` of a slice from [buffer] according to [reduce] and [axis].
num reduceMeanSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num sumValue = dType.isInt ? 0 : 0.0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    sumValue += buffer[indexForTensor];
  }
  return dType.isInt ? sumValue ~/ sliceSize : sumValue / sliceSize;
}

/// Computes the `product` of a slice from [buffer] according to [reduce] and [axis].
num reduceProdSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num prodValue = dType.isInt ? 1 : 1.0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    prodValue *= buffer[indexForTensor];
  }
  return prodValue;
}

/// Computes the `variance` of a slice from [buffer] according to [reduce] and [axis].
num reduceVarianceSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num sumValue = dType.isInt ? 0 : 0.0;
  num squareSumValue = dType.isInt ? 0 : 0.0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    sumValue += buffer[indexForTensor];
    squareSumValue += buffer[indexForTensor] * buffer[indexForTensor];
  }
  return dType.isInt
      ? (squareSumValue - (sumValue * sumValue / sliceSize)) ~/ sliceSize
      : (squareSumValue - (sumValue * sumValue / sliceSize)) / sliceSize;
}

/// Computes the `standard deviation` of a slice from [buffer] according to [reduce] and [axis].
num reduceStdSlice(List buffer, List<int> reduce, List<int> cumDimsSize, List<int> axis, DType dType) {
  final List<int> shape = List.generate(reduce.length, (i) => axis.contains(i) ? reduce[i] : 1);
  final sliceSize = shape.reduce((e1, e2) => e1 * e2);
  num sumValue = dType.isInt ? 0 : 0.0;
  num squareSumValue = dType.isInt ? 0 : 0.0;
  List<int> currentIndices = List<int>.filled(shape.length, 0);
  int indexForTensor = 0;
  for (int i = 0; i < sliceSize; i += 1) {
    indexForTensor = 0;
    int index = i;

    for (int j = shape.length - 1; j >= 0; j -= 1) {
      currentIndices[j] = index % shape[j];
      index = index ~/ shape[j];
    }

    for (int k = 0; k < shape.length; k += 1) {
      indexForTensor += cumDimsSize[k] * (axis.contains(k) ? currentIndices[k] : reduce[k]);
    }

    sumValue += buffer[indexForTensor];
    squareSumValue += buffer[indexForTensor] * buffer[indexForTensor];
  }
  return dType.isInt
      ? sqrt((squareSumValue - (sumValue * sumValue / sliceSize)) / sliceSize).toInt()
      : sqrt((squareSumValue - (sumValue * sumValue / sliceSize)) / sliceSize);
}
