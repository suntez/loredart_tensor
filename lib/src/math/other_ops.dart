import 'package:loredart_tensor/src/utils/dtype_utils.dart';

import '../tensors/int32list_num_tensor.dart';
import '../tensors/tensor.dart';

/// Concatenates [Tensor]s of [tensors] along one given dimension [axis].
/// 
/// Elements of [tensors] must be [NumericTensor]s with same [DType], same rank and have equal shapes, except in the [axis] index.
/// 
/// If [tensors[i].shape] == `[d0, d1, ..., Daxis(i), ..., dn]`, then concatenated tensor has shape `[d0, d1, ..., R, ..., dn]`,
/// where `R = sum(Daxis(i))`.
/// 
/// Returns [Tensor] with the same [DType] as elements of [tensors].
/// 
/// Example:
/// ```dart
/// Tensor x1 = Tensor.fill([1,2,1], 1.0);
/// // [[[1.0]
/// //  [1.0]]]
/// Tensor x2 = Tensor.fill([1,2,2], 2.0);
/// // [[[2.0, 2.0]
/// //  [2.0, 2.0]]]
/// 
/// Tensor c = concat([x1, x2], axis: -1);
/// print(c);
/// // <Tensor(shape: [1, 2, 3], values:
/// //  [[[1.0, 2.0, 2.0]
/// //   [1.0, 2.0, 2.0]]], dType: float32)>
/// ```
Tensor concat(List<Tensor> tensors, {int axis = - 1}) {
  if (tensors.isEmpty || tensors.any((element) => element is! NumericTensor || element.dType != tensors[0].dType || element.rank != tensors[0].rank)) {
    throw ArgumentError('Expect non-empty list of NumericTensors with same DType and same rank', 'tensors');
  }
  if (axis >= tensors[0].rank || axis < -tensors[0].rank) {
    throw ArgumentError('axis must be in halp-interval [-rank, +rank), but reiceved $axis', 'axis');
  }
  axis %= tensors[0].rank;

  final List<List<int>> shapes = List.generate(tensors.length, (i) => List.from(tensors[i].shape.list)..removeAt(axis));
  for (int i = 0; i < tensors[0].rank-1; i += 1) {
    if (shapes.any((element) => element[i] != shapes[0][i])) {
      throw ArgumentError('All dimensions in tensors shapes except axis must be equal', 'tensors');
    }
  }

  List<int> cumSum = [tensors[0].shape[axis]];
  for (int i = 0; i < tensors.length - 1; i += 1) {
    cumSum.add(cumSum[i] + tensors[i+1].shape[axis]);
  }

  List<int> shape = List.from(tensors[0].shape.list);
  shape[axis] = cumSum[tensors.length-1];
  List buffer = emptyBuffer(tensors[0].dType, shape.reduce((e1, e2) => e1*e2));
  final int beforeAxisSize = axis == 0 ? 1 : shape.sublist(0, axis).reduce((e1, e2) => e1*e2);
  final int afterAxisSize = axis == tensors[0].rank-1 ? 1 : shape.sublist(axis+1).reduce((e1, e2) => e1*e2);

  for (int b = 0; b < beforeAxisSize; b += 1) {
    for (int a = 0; a < afterAxisSize; a += 1) {
      for (int i = 0; i < shape[axis]; i += 1) {
        int tensorIndex = 0;
        for (int k = 0; k < cumSum.length; k += 1) {
          if (i < cumSum[k]) {
            tensorIndex = k;
            break;
          }
        }

        buffer[b*shape[axis]*afterAxisSize + i * afterAxisSize + a] = (tensors[tensorIndex] as NumericTensor)
          .buffer[
            b * tensors[tensorIndex].shape[axis] * afterAxisSize + (tensorIndex == 0 ? i : (i - cumSum[tensorIndex-1])) * afterAxisSize + a];
      }
    }
  }
  return Tensor.fromBuffer(buffer, shape, dType: tensors[0].dType);
}

/// Extracts and returns strode slice of the tensor [x]
/// from [begin] (inclusively) to [end] (exclusively).
/// 
/// [begin] and [end] must have length of [x.rank].
/// 
/// Slicing doesn't support negative indices, so req `0 <= start[i] <= end[i]` must be meet,
/// otherwise will throw an ArgumentError.
/// 
/// Returns a [Tensor] of the same [DType] as [x].
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.constant([
///     [[1,1,1], [2,2,2], [3,3,3]], [[4,4,4], [5,5,5], [6,6,6]]
///  ]); // shape [2,2,3]
/// 
/// slice(x, [0,0,0], [1,2,3]);
/// // <Tensor(shape: [1, 2, 3], values:
/// //  [[[1 1 1]
/// //   [2 2 2]]], dType: int32)>
/// 
/// slice(x, [1,0,0], [1,3,1]);
/// // <Tensor(shape: [3, 1], values:
/// //  [[4][5][6]], dType: int32)>
/// ```
Tensor slice(Tensor x, List<int> begin, List<int> end) {
  if (begin.length != x.rank || end.length != x.rank) {
    throw ArgumentError('begin and end must have length = x.rank, but reiceved begin.length: ${begin.length}, end.length: ${end.length}, x.rank: ${x.rank}');
  }
  for (int i = 0; i < x.rank; i += 1) {
    if (begin[i] < 0 || end[i] < 0) {
      throw ArgumentError("Slicing doesn't support negative indices in the begin and the end of slice, but reiceved: $begin - $end");
    }
    if (begin[i] > end[i] || end[i] > x.shape[i]) {
      throw ArgumentError("Invalid values of slice: begin[i] should be in [0, end[i]], while end[i] in [begin[i], x.shape[i]], but reiceved $begin - $end, with tensor shape: ${x.shape}");
    }
  }
  if (x is NumericTensor) {
    final List<int> cumProd = List<int>.generate(x.rank, (i) => i == x.rank-1 ? 1 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2));
    final List<int> shape = List.generate(x.rank, (i) => end[i] - begin[i] == 0 ? 1 : end[i] - begin[i]);
    final List<int> shapeWith0 = List.generate(x.rank, (i) => end[i] - begin[i]);
    List buffer = emptyBuffer(x.dType, shape.reduce((e1, e2) => e1*e2));

    List<int> currentIndices = List<int>.filled(shape.length, 0);
    int indexForTensor = 0;
    for (int i = 0; i < buffer.length; i += 1) {
      indexForTensor = 0;
      int index = i;
      for (int j = shape.length - 1; j >= 0; j -= 1) {
        currentIndices[j] = index % shape[j];
        index = index ~/ shape[j];
      }
      for (int k = 0; k < shape.length; k += 1) {
        indexForTensor += cumProd[k] * (currentIndices[k] + begin[k]);
      }
      buffer[i] = x.buffer[indexForTensor];
    }
    return Tensor.fromBuffer(buffer, shapeWith0.where((element) => element != 0).toList(), dType: x.dType);
  } else {
    throw UnimplementedError('Slicing is not supported for non-numeric tensors, but reiceved ${x.runtimeType}');
  }
}

/// Pads tensor [x] with [value] according to the [padding].
/// 
/// [padding] is a [Int32NumericTensor] (Tensor with DType.int32) of shape `[n,2]`, where n = [x.rank].
/// 
/// `padding[i, 0]` indicates how indicates how many values to add before the contents of tensor in i-th dimension,
/// and `padding[i, 1]` indicates how many values to add after the contents of tensor in i-th dimension.
/// 
/// Returns a [Tensor] of the same [DType] as [x] with i-th dims equal `padding[i, 0] + x.shape[i] + padding[i, 1]`.
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.fill([2,3], 2, dType: int64); // [[2, 2, 2]
///                                                       // [2, 2, 2]]
/// Tensor padding = Tensor.constant([[1,2], [2,1]]);
/// 
/// pad(x, padding);
/// // <Tensor(shape: [5, 6], values:
/// // [[0, 0, 0, 0, 0, 0]
/// //  [0, 0, 2, 2, 2, 0]
/// //  [0, 0, 2, 2, 2, 0]
/// //  [0, 0, 0, 0, 0, 0]
/// //  [0, 0, 0, 0, 0, 0]], dType: int64)>
/// ```
Tensor pad(Tensor x, Tensor padding, {num value = 0.0}) {
  if (x is NumericTensor && padding is Int32NumericTensor) {
    if (padding.rank == 2 && padding.shape[0] == x.rank && padding.shape[1] == 2) {
      if (padding.buffer.any((element) => element < 0)) {
        throw ArgumentError('Only non-negative values are acceptable', 'padding');
      }
      value = x.dType.isInt ? value.toInt() : value.toDouble();
      List<int> shape = List.generate(x.rank, (i) => (x.shape[i] + padding.buffer[i*2 + 0] + padding.buffer[i*2 + 1]));
      final size = shape.reduce((e1, e2) => e1*e2);
      List buffer = emptyBuffer(x.dType, size);

      List<int> currentIndices = List<int>.filled(shape.length, 0);
      final List<int> cumProd = List<int>.generate(shape.length, (i) => i == shape.length-1 ? 1 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2));
      int indexFromX = 0;
      bool padCurrent = false;
      
      for (int i = 0; i < size; i += 1) {
        indexFromX = 0;
        padCurrent = false;
        
        int index = i;
        for (int j = shape.length - 1; j >= 0; j -= 1) {
          currentIndices[j] = index % shape[j];
          index = index ~/ shape[j];
        }

        for (int k = 0; k < shape.length; k += 1) {
          if ((currentIndices[k] < padding.buffer[k*2+0]) || ( (currentIndices[k] - padding.buffer[k*2+0]) >= x.shape[k])) {
            padCurrent = true;
            break;
          }
        }
        if (padCurrent) {
          buffer[i] = value;
        } else {
          for (int k = 0; k < shape.length; k += 1) {
            indexFromX += cumProd[k] * (currentIndices[k] - padding.buffer[k*2+0]);
          }
          buffer[i] = x.buffer[indexFromX];
        }
      }
      return Tensor.fromBuffer(buffer, shape, dType: x.dType);
    } else {
      throw ArgumentError('Expected Int32NumericTensor with rank 2 and shape [n, 2], where n == x.rank', 'padding');
    }
  } else {
    throw ArgumentError('Expected NumericTensors as input x and Int32NumericTensor as padding, but reiceved x: ${x.runtimeType} and padding: ${padding.runtimeType}');
  }
}

/// Reshapes a tensor [x] with new [shape].
/// 
/// Returns [Tensor] that has the same values as [x] in the same order and same [DType] but with new [shape].
/// 
/// Throws ArgumentError if size of new [shape] is not equal [x.shape.size].
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.constant([2,4,6,8,10,12], shape: [3,2]);
/// // [[2, 4]
/// //  [6, 8]
/// //  [10, 12]]
/// 
/// reshape(x, [2,1,3]);
/// // <Tensor(shape: [2, 1, 3], values:
/// // [[[2 4 6]]
/// // [[8 10 12]]], dType: int32)>
/// ```
Tensor reshape(Tensor x, List<int> shape) {
  if (x.shape.size == shape.reduce((e1, e2) => e1*e2)) {
    if (x is NumericTensor) {
      return Tensor.fromBuffer(x.buffer, shape, dType: x.dType);
    } else {
      throw ArgumentError('Reshape is not supported for ${x.runtimeType}');
    }
  } else {
    throw ArgumentError('Cannot reshape Tensor with ${x.shape.size} elements into shape $shape', 'shape');
  }
}

/// Casts [x] to a [dType].
/// 
/// Throws an ArgumentError if [dType] is non-numeric.
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3]);
/// // <Tensor(shape: [3], values:
///  [[1, 2, 3]], dType: int32)>
/// 
/// cast(x, DType.float64);
/// // <Tensor(shape: [3], values:
/// // [[1.0, 2.0, 3.0]], dType: float64)>
/// ```
Tensor cast(Tensor x, DType dType) {
  if (x is NumericTensor) {
    if (!dType.isNumeric) {
      throw ArgumentError('Cannot cast NumericTensor into non-numeric ($dType)', 'dType');
    }
    List buffer = emptyBuffer(dType, x.shape.size);
    for (int i = 0; i < x.shape.size; i += 1) {
      buffer[i] = dType.isInt ? (x.buffer[i] as num).toInt() : (x.buffer[i] as num).toDouble();
    }
    return Tensor.fromBuffer(buffer, x.shape.list, dType: dType);
  } else {
    throw UnimplementedError('Casting is not supported for ${x.runtimeType}');
  }
}

/// Adds extra dim of length 1 into [x] at index [axis].
/// 
/// Operation support negative indexes for [axis].
/// 
/// Returns [Tensor] of the same [DType] as [x].
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.ones([2,3,4]);
/// expandDims(x, -1).shape; // [2, 3, 4, 1]
/// expandDims(x, -2).shape; // [2, 3, 1, 4]
/// expandDims(x, 0).shape; // [1, 2, 3, 4]
/// expandDims(x, 1).shape; // [2, 1, 3, 4]
/// ```
Tensor expandDims(Tensor x, int axis) {
  if (x is NumericTensor) {
    if (axis >= x.rank || axis < -x.rank) {
      throw ArgumentError('axis must be in the interval [-x.rank, +x.rank), but reiceved $axis', 'axis');
    }
    axis %= x.rank;
    return Tensor.fromBuffer(x.buffer, List.from(x.shape.list)..insert(axis, 1), dType: x.dType);
  } else {
    throw ArgumentError('Expected NumericTensor as x, but reiceved ${x.runtimeType}', 'x');
  }
}

//TODO: update API for `axis` parameter
/// Removes dimensions of size 1 from the shape of a [x].
/// 
/// Return [Tensor] of the same [DType] of [x].
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.ones([2,1,3,1,4]);
/// 
/// squeeze(x).shape; // [2, 3, 4]
/// ```
Tensor squeeze(Tensor x, {List<int>? axis}) {
  axis ??= List.generate(x.rank, (i) => i);
  if (axis.any((e) => e <= -x.rank || e > x.rank)) {
    throw ArgumentError();
  }
  axis = axis.map((e) => e%x.rank).toList();
  if (x is NumericTensor) {
    final shape = [for (int i = 0; i < x.rank; i += 1) if (!(axis.contains(i) && x.shape[i] == 1)) x.shape[i]];
    return Tensor.fromBuffer(x.buffer, shape, dType: x.dType);
  } else {
    throw ArgumentError('Expected NumericTensor as x, but reiceved ${x.runtimeType}', 'x');
  }
}


Tensor oneHotTensor(Tensor indices, {required int depth, num onValue = 1.0, num offValue = 0.0, int axis = -1, DType? dType}) {
  if (!indices.dType.isInt) {
    throw ArgumentError('Indices tensor should be integer base, but received tensor of ${indices.dType}', 'indices');
  }
  if (dType != null) {
    if ((dType.isInt && (onValue is! int || offValue is! int)) || (dType.isDouble && (onValue is! double || offValue is! double))) {
      throw ArgumentError('On- and off- values must be of the same type as specified dType, but received dType: $dType, onValue type: ${onValue.runtimeType}, offValue type: ${offValue.runtimeType}');
    }
  } else {
    if (onValue.runtimeType != offValue.runtimeType) {
      throw ArgumentError('On- and off- values must be of the same type, but received onValue type: ${onValue.runtimeType}, offValue type: ${offValue.runtimeType}');
    } else {
      dType = onValue is int ? DType.int32 : DType.float32;
    }
  }
  if (depth <= 0) {
    throw ArgumentError('Depth must be a positive integer, but received $depth', 'depth');
  }
  if (axis < -1 || axis > indices.rank) {
    throw ArgumentError('The axis should be between [-1, indices.rank), but received $axis', 'axis');
  }

  List<int> outputShape = List<int>.from(indices.shape.list);
  if (axis == -1) {
    axis = indices.rank;
  }
  outputShape.insert(axis, depth);
  print(axis);
  print(outputShape);

  List buffer = emptyBuffer(dType, outputShape.reduce((e1,e2)=>e1*e2));

  int sizeBeforeAxis = outputShape.sublist(0, axis).reduce((e1, e2) => e1*e2);
  print('bef: $sizeBeforeAxis');
  int sizeAfterAxis = axis == indices.rank ? 1 : outputShape.sublist(axis).reduce((e1, e2) => e1*e2);
  print('bef: $sizeAfterAxis');
  indices as NumericTensor;
  for (int b = 0; b < sizeBeforeAxis; b += 1) {
    for (int a = 0; a < sizeAfterAxis; a += 1) {
      for (int i = 0; i < depth; i += 1) {
        if (indices.buffer[b*sizeAfterAxis + a] == i) {
          buffer[(b*depth + i)*sizeAfterAxis + a] = onValue;
        } else {
          buffer[(b*depth + i)*sizeAfterAxis + a] = offValue;
        }
      }
    }
  }
  return Tensor.fromBuffer(buffer, outputShape, dType: dType);
}