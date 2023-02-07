import 'package:loredart_tensor/src/utils/dtype_utils.dart';

import '../tensors/num_tensor.dart';
import '../tensors/tensor.dart';
import 'reduce_slices.dart';

/// Computes [reduceRule] function of elements across dimensions given in [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes [reduceRule] over all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis].
/// 
/// The [DType] of resulting tensor might be specify in [dType].
Tensor _reduceTensor(Tensor x, num Function(List, List<int>, List<int>, List<int>, DType) reduceRule, {List<int>? axis, bool keepDims = false, DType? dType}) {
  if (x is NumericTensor) {
    final List<int> cumProd = List<int>.generate(x.rank, (i) => i == x.rank-1 ? (x.shape[i] == 1 ? 0 : 1) : (x.shape[i] == 1 ? 0 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
    dType ??= x.dType;
    if (axis == null || axis.isEmpty) {
      return Tensor.constant([reduceRule(x.buffer, x.shape.list, cumProd, List.generate(x.rank, (i) => i), dType)]);
    } else {

      for (int i = 0; i < axis.length; i += 1) {
        if (axis[i] >= x.rank || axis[i] <= -x.rank) {
          throw RangeError.value(axis[i], 'axis', 'Axis should include elements from interval [-x.rank, +x.rank), but got axis[$i] = ${axis[i]}');
        }
        axis[i] %= x.rank;
      }

      if (axis.length > x.rank || axis.length < axis.toSet().length) {
        throw ArgumentError('Axis.length mast be less or equal x.rank and must contain unique elements, but got $axis', 'axis');
      }
      List<int> shape = List.generate(x.rank, (i) => axis.contains(i) ? 1 : x.shape[i]);
      if (shape.every((element) => element == 1)) {
        return Tensor.constant([reduceRule(x.buffer, x.shape.list, cumProd, List.generate(x.rank, (i) => i), dType)]);
      } else {
        List buffer = emptyBuffer(dType, shape.reduce((e1, e2) => e1 * e2));
        List<int> currentIndices = List<int>.filled(shape.length, 0);
        for (int i = 0; i < buffer.length; i += 1) {
          int index = i;
          for (int j = shape.length - 1; j >= 0; j -= 1) {
            currentIndices[j] = index % shape[j];
            index = index ~/ shape[j];
          }
          buffer[i] = reduceRule(x.buffer, List.generate(x.rank, (i) => shape[i] == 1 ? x.shape[i] : currentIndices[i]), cumProd, axis, dType);
        }
        if (!keepDims) {
          shape = [for (int i = 0; i < shape.length; i += 1) if (!(shape[i] == 1 && axis.contains(i))) shape[i]];
        }
        return Tensor.fromTypedDataList(buffer, shape, dType: dType);
      }
    }
  } else {
    throw ArgumentError('Expected NumericalTensor, but got ${x.runtimeType}', 'x');
  } 
}

/// Computes max of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes max of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x].
/// (Like [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max),
/// `reduceMax` has an aggressive type inference from [x]).
/// 
/// `Note`: Prefer to use a `maximum` function over `reduceMax` with [axis]: null, if you need to find the maximum of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// 
/// reduceMax(x); // <Tensor(shape: [1], values: [[6]], dType: int32)>
/// reduceMax(x, axis: [-1]); // <Tensor(shape: [2], values: [[3, 6]], dType: int32)>
/// reduceMax(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[4, 5, 6]], dType: int32)>
/// ```
Tensor reduceMax(Tensor x, {List<int>? axis, bool keepDims = false}) => _reduceTensor(x, reduceMaxSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes min of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x].
/// (Like [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_min),
/// `reduceMin` has an aggressive type inference from [x]).
/// 
/// `Note`: Prefer to use a `minimum` function over `reduceMin` with [axis]: null, if you need to find the minimum of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceMin(x); // <Tensor(shape: [1], values: [[1]], dType: int32)>
/// reduceMin(x, axis: [-1]); // <Tensor(shape: [2], values: [[1, 4]], dType: int32)>
/// reduceMin(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[1, 2, 3]], dType: int32)>
/// ```
Tensor reduceMin(Tensor x, {List<int>? axis, bool keepDims = false}) => _reduceTensor(x, reduceMinSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes the sum of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes the sum of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x].
/// (Like [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum),
/// `reduceSum` has an aggressive type inference from [x]).
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceSum(x); // <Tensor(shape: [1], values: [[21]], dType: int32)>
/// reduceSum(x, axis: [-1]); // <Tensor(shape: [2], values: [[6, 15]], dType: int32)>
/// reduceSum(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[5, 7, 9]], dType: int32)>
/// ```
Tensor reduceSum(Tensor x, {List<int>? axis, bool keepDims = false}) => _reduceTensor(x, reduceSumSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes the product of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes the product of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x].
/// (Like [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod),
/// `reduceProd` has an aggressive type inference from [x]).
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceProd(x); // <Tensor(shape: [1], values: [[720]], dType: int32)>
/// reduceProd(x, axis: [-1]); // <Tensor(shape: [2], values: [[6, 120]], dType: int32)>
/// reduceProd(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[4, 10, 18]], dType: int32)>
/// ```
Tensor reduceProd(Tensor x, {List<int>? axis, bool keepDims = false}) => _reduceTensor(x, reduceProdSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes the mean of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes the mean of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the specified [dType], by default its inherited from [x].
/// (Unlike [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean),
/// `reduceMean` has an non-aggressive type inference from [x], and behaves more like `numpy` function).
/// 
/// `Note`: Prefer to use a `mean` function over `reduceMean` with [axis]: null, if you need to find the mean of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: float32); // [[1.0, 2.0, 3.0]
///                                                                          //  [4.0, 5.0, 6.0]]
/// 
/// reduceMean(x); // <Tensor(shape: [1], values: [[3.5]], dType: float32)>
/// reduceMean(x, axis: [-1], dType: int32); // <Tensor(shape: [2], values: [[2, 5]], dType: int32)>
/// reduceMean(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[2.5, 3.5, 4.5]], dType: float32)>
/// ```
Tensor reduceMean(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => _reduceTensor(x, reduceMeanSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);

/// Computes the variance of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes the variance of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the specified [dType], by default its inherited from [x].
/// (Unlike [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance),
/// `reduceVariance` has an non-aggressive type inference from [x], and behaves more like `numpy` function).
/// 
/// `Note`: Prefer to use a `variance` function over `reduceVariance` with [axis]: null, if you need to find the variance of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: DType.float32); // [[1.0, 2.0, 3.0]
///                                                                                //  [4.0, 5.0, 6.0]]
/// 
/// reduceVariance(x); // <Tensor(shape: [1], values: [[2.9166...]], dType: float32)>
/// reduceVariance(x, axis: [-1]); // <Tensor(shape: [2], values: [[0.66..., 0.66...]], dType: float32)>
/// reduceVariance(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[2.25, 2.25, 2.25]], dType: float32)>
/// ```
Tensor reduceVariance(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => _reduceTensor(x, reduceVarianceSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);

/// Computes the standard deviation of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes the standard deviation of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the specified [dType], by default its inherited from [x].
/// (Unlike [corresponding function from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/reduce_std),
/// `reduceStd` does not has an aggressive type inference from [x], and behaves more like `numpy` function).
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: DType.float32); // [[1.0, 2.0, 3.0]
///                                                                          //  [4.0, 5.0, 6.0]]
/// 
/// reduceStd(x); // <Tensor(shape: [1], values: [[1.707...]], dType: float32)>
/// reduceStd(x, axis: [-1]); // <Tensor(shape: [2], values: [[0.816..., 0.816...]], dType: float32)>
/// reduceStd(x, axis: [0], keepDims: true); // <Tensor(shape: [1, 3], values: [[1.5, 1.5, 1.5]], dType: float32)>
/// ```
Tensor reduceStd(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => _reduceTensor(x, reduceStdSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);


/// Returns the `local indices` with the largest elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// but all entries must be unique, otherwise will throw an ArgumentError.
/// 
/// If [axis.length] is 1 the function behaves as `argMax`, but keeps the shape of the Tensor.
/// In other cases, function will combine the dimensions in [axis] and return the indices within new dim.
/// 
/// It's better to use the `argMax` function, because that one was meant to be used in specific calculations. 
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the specified integer based [dType], by default its [DType.int32].
/// 
/// Examples:
/// ```dart
/// final x = Tensor.constant([10,2,3,4,
///                            20,1,7,8,
///                            12,5,13,9], shape: [2,3,2]);
/// 
/// reduceLocalArgMax(x, axis: [-1]); // <Tensor(shape: [2, 3], values: [[0, 1, 0], [1, 0, 0]], dType: int32)>
/// 
/// reduceLocalArgMax(x, axis: [0, 1]); // <Tensor(shape: [2], values: [[2, 5]], dType: int32)>
/// 
/// reduceLocalArgMax(x, axis: [0, -1]); // <Tensor(shape: [3], values: [[0, 2, 0]], dType: int32)>
/// 
/// reduceLocalArgMax(x, axis: [1, -1]); // <Tensor(shape: [2], values: [[4, 4]], dType: int32)>
/// ```
Tensor reduceLocalArgMax(Tensor x, {List<int>? axis, bool keepDims = false}) => _reduceTensor(x, reduceLocalArgMaxSlice, axis: axis, keepDims: keepDims, dType: DType.int32);


/// Returns the indices with the largest elements across dimension given in the [axis] of a tensor [x].
/// 
/// [axis] may be from half-interval [-x.rank, x.rank) and by default is equal to 0.
/// 
/// Returns reduces [Tensor] of the specified integer based [dType], by default its [DType.int32].
/// 
/// Examples:
/// ```dart
/// final x = Tensor.constant([10,2,3,4,20,1], shape: [2,3]); // [[10, 2, 3]
///                                                           //  [4, 20, 1]
/// 
/// argMax(x); // <Tensor(shape: [3], values: [[0, 1, 0]], dType: int32)>
/// argMax(x, axis: -1); // <Tensor(shape: [2], values: [[0, 1]], dType: int32)>
/// ```
Tensor argMax(Tensor x, {int axis = 0, DType dType = DType.int32}) {
  if (!dType.isInt) {
    throw ArgumentError("Only excepts the integer DTypes, but received $dType", 'dType');
  }
  return _reduceTensor(x, reduceLocalArgMaxSlice, axis: [axis], keepDims: false, dType: dType);
}