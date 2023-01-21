import 'package:loredart_tensor/src/utils/dtype_utils.dart';

import '../tensors/tensor.dart';
import 'reduce_slices.dart';

/// Computes [reduceRule] function of elements across dimensions given in [axis](es) of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes [reduceRule] over all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis].
/// 
/// The [DType] of resulting tensor might be specify in [dType].
Tensor reduceTensor(Tensor x, num Function(List, List<int>, List<int>, List<int>, DType) reduceRule, {List<int>? axis, bool keepDims = false, DType? dType}) {
  if (x is NumericTensor) {
    final List<int> cumProd = List<int>.generate(x.rank, (i) => i == x.rank-1 ? 1 : (x.shape[i] == 1 ? 0 : x.shape.list.sublist(i+1).reduce((e1, e2) => e1*e2)));
    dType ??= x.dType;
    if (axis == null || axis.isEmpty) {
      return Tensor.constant([reduceRule(x.buffer, x.shape.list, cumProd, List.generate(x.rank, (i) => i), dType)]);
    } else {

      for (int i = 0; i < axis.length; i += 1) {
        if (axis[i] >= x.rank || axis[i] <= -x.rank) {
          throw ArgumentError('Axis should include elements from interval [-x.rank, +x.rank), but got axis[$i] = ${axis[i]}', 'axis');
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
        return Tensor.fromBuffer(buffer, shape, dType: dType);
      }
    }
  } else {
        throw ArgumentError(
        'Expected NumericalTensor, but got ${x.runtimeType}', 'x');
  } 
}

/// Computes max of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes max of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Like corresponding function from `TensorFlow`, `reduceMax` has an aggressive type inference from [x]).
/// 
/// `Note`: Prefer to use a `maximum` function over `reduceMax` with [axis: null], if you need to find the maximum of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceMax(x);
/// // <Tensor(shape: [1], values: [[6]], dType: int32)>
/// reduceMax(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[3, 6]], dType: int32)>
/// reduceMax(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[4, 5, 6]], dType: int32)>
/// ```
Tensor reduceMax(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceMaxSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes min of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Like corresponding function from `TensorFlow`, `reduceMin` has an aggressive type inference from [x]).
/// 
/// `Note`: Prefer to use a `minimum` function over `reduceMin` with [axis: null], if you need to find the minimum of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceMin(x);
/// // <Tensor(shape: [1], values: [[1]], dType: int32)>
/// reduceMin(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[1, 4]], dType: int32)>
/// reduceMin(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[1, 2, 3]], dType: int32)>
/// ```
Tensor reduceMin(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceMinSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes sum of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Like corresponding function from `TensorFlow`, `reduceSum` has an aggressive type inference from [x]).
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceSum(x);
/// // <Tensor(shape: [1], values: [[21]], dType: int32)>
/// reduceSum(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[6, 15]], dType: int32)>
/// reduceSum(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[5, 7, 9]], dType: int32)>
/// ```
Tensor reduceSum(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceSumSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes product of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Like corresponding function from `TensorFlow`, `reduceProd` has an aggressive type inference from [x]).
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3]); // [[1,2,3]
///                                                          //  [4,5,6]]
/// reduceProd(x);
/// // <Tensor(shape: [1], values: [[720]], dType: int32)>
/// reduceProd(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[6, 120]], dType: int32)>
/// reduceProd(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[4, 10, 18]], dType: int32)>
/// ```
Tensor reduceProd(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceProdSlice, axis: axis, keepDims: keepDims, dType: x.dType);

/// Computes mean of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Unlike corresponding function from `TensorFlow`,
/// `reduceMean` has a non-aggressive type inference from [x],
/// and [dType] of output can be specified explicitly). By default [dType] is inherited from [x].
/// 
/// `Note`: Prefer to use a `mean` function over `reduceMean` with [axis: null], if you need to find the mean of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: float32);
/// // [[1.0, 2.0, 3.0]
/// //  [4.0, 5.0, 6.0]]
/// 
/// reduceMean(x);
/// // <Tensor(shape: [1], values: [[3.5]], dType: float32)>
/// reduceMean(x, axis: [-1], dType: int32);
/// // <Tensor(shape: [2], values: [[2, 5]], dType: int32)>
/// reduceMean(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[2.5, 3.5, 4.5]], dType: float32)>
/// ```
Tensor reduceMean(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => reduceTensor(x, reduceMeanSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);

/// Computes variance of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Unlike corresponding function from `TensorFlow`,
/// `reduceVariance` has a non-aggressive type inference from [x],
/// and [dType] of output can be specified explicitly). By default [dType] is inherited from [x].
/// 
/// `Note`: Prefer to use a `variance` function over `reduceVariance` with [axis: null], if you need to find a variance of [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: DType.float32);
/// // [[1.0, 2.0, 3.0]
/// //  [4.0, 5.0, 6.0]]
/// 
/// reduceVariance(x);
/// // <Tensor(shape: [1], values: [[2.9166...]], dType: float32)>
/// reduceVariance(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[0.66..., 0.66...]], dType: float32)>
/// reduceVariance(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[2.25, 2.25, 2.25]], dType: float32)>
/// ```
Tensor reduceVariance(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => reduceTensor(x, reduceVarianceSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);

/// Computes standard deviation of elements across dimensions given in the [axis] of a tensor [x].
/// 
/// [axis] may include indices from half-interval [-x.rank, x.rank) in any order,
/// all entries must be unique, otherwise will throw an ArgumentError.
/// If [axis] is `null` computes min of all elements of [x].
/// 
/// Unless [keepDims] is `false`, the rank of the tensor is reduced by 1 for each of the entries in [axis],
/// but if [keepDims] is `true`, the reduced dimensions are retained with length 1.
/// 
/// Returns reduces [Tensor] of the same [DType] as [x]
/// (Unlike corresponding function from `TensorFlow`,
/// `reduceStd` has a non-aggressive type inference from [x],
/// and [dType] of output can be specified explicitly). By default [dType] is inherited from [x].
/// 
/// Examples:
/// ```dart
/// Tensor x = Tensor.constant([1,2,3,4,5,6], shape: [2,3], dType: DType.float32);
/// // [[1.0, 2.0, 3.0]
/// //  [4.0, 5.0, 6.0]]
/// 
/// reduceStd(x);
/// // <Tensor(shape: [1], values: [[1.707...]], dType: float32)>
/// reduceStd(x, axis: [-1]);
/// // <Tensor(shape: [2], values: [[0.816..., 0.816...]], dType: float32)>
/// reduceStd(x, axis: [0], keepDims: true);
/// // <Tensor(shape: [1, 3], values: [[1.5, 1.5, 1.5]], dType: float32)>
/// ```
Tensor reduceStd(Tensor x, {List<int>? axis, bool keepDims = false, DType? dType}) => reduceTensor(x, reduceStdSlice, axis: axis, keepDims: keepDims, dType: dType ?? x.dType);

// TODO: API for argmax
Tensor reduceGlobalArgMax(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceGlobalArgMaxSlice, axis: axis, keepDims: keepDims, dType: DType.int32);


Tensor reduceLocalArgMax(Tensor x, {List<int>? axis, bool keepDims = false}) => reduceTensor(x, reduceLocalArgMaxSlice, axis: axis, keepDims: keepDims, dType: DType.int32);