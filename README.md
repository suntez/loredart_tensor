A package for creation and manipulation with multidimensional data arrays in the form of a `Tensor` class.

The package is somehow a Dart implementation of Tensor API from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/Tensor) (or/and NumPy library). And was initially created to power the [loredart_nn](https://pub.dev/packages/loredart_nn) package, which is also inspired by the TF (and currently is under rebuilding).

## The `Tensor`s ![](assets/tesseractMini.png)

Each `Tensor` is described by two main properties:

- shape: `TensorShape` instance
- single data type: `DType` instance

The `DType` of the `Tensor` determines the type of the values and how `loredart` stores them. For now, the only supported `DType`s are the numerical ones (`float32`, `int64` etc.).

There are many ways to create a Tensor instance, the basic ones - named factories of class:

```dart
final x = Tensor.constant([2,0,2,3], shape: [2,2]);
print(x);
// <Tensor(shape: [2, 2], values:
// [[2, 0]
// [2, 3]], dType: int32)>

final y = Tensor.fill([3,3], 0.3, dType: DType.float64);
print(y);
// <Tensor(shape: [3, 3], values:
//  [[0.3, 0.3, 0.3]
//  [0.3, 0.3, 0.3]
//  [0.3, 0.3, 0.3]], dType: float64)>
```

There is no explicit limitation on the Tensors rank (length of the shape):

```dart
final x = Tensor.ones([1,1,2,3,5,8,13,21,34], dType: DType.float64);

print(x.shape); // [1,1,2,3,5,8,13,21,34]

print(x.shape[-3]); // 13

// Number of indices to get a single element
print(x.rank); // 9

// Number of elements in the tensor
print(x.shape.size); // 2227680
```

Also, there are some particular ways to create tensor, like random generation:

```dart
final x = uniform([10, 10], min: -3, max: 0);
print(x.dType); // DType.float32
print(x.shape); // [10, 10]
```
And many others, e.g., `oneHotTensor`, `Tensor.fill`, `Tensor.zeros`, etc.

Like TensorFlow tensors, `loredart` ones are immutable (with final fields), and any operation on `Tensor`(s) will produce a new instance.

## Operations on `Tensor`s

There are many implemented and documented operations on Tensors, including math and linear algebra funcs, shape transformations, casting, reducing operations, etc.

The operation requires Tensors of the same DType, and most of them - require equal shapes; however, there are some [exceptions](#arithmetic-and-comparison).

### Arithmetic and comparison ops

Arithmetic and comparison operations support broadcasting, which means they can operate on the `Tensor`s with broadcastable shapes or even with Dart `num`s.

```dart
final x = Tensor.ones([3,4,5]); // with DType.float32

final y = Tensor.ones([3,1,5]);
print((x+y).shape); // [3, 4, 5]

final v = Tensor.ones([4,5]);
print((x*y).shape); // [3, 4, 5]

final k = Tensor.ones([1,1,1,1,1]);
print((x/k).shape); // [1, 1, 3, 4, 5]

print(less(x, 3.5).shape); // [3, 4, 5]
```

### Math and statistics

`loredart` supports most of the math and statistical functions:
```dart
final x = Tensor.constant(<double>[0,1,2,3,4,5], shape: [2, 3]);
var t = exp(x); // element-wise exp
print(t.shape); // same as x.shape

var m = mean(x); // the mean of the whole tensor
print(m.shape); // [1]

final y = Tensor.diag([3, 4], numCols: 3); // shape is [2, 3]
pow(x, y); // element-wise pow
```
, but more interesting are the reducing versions of them:
```dart
final x = Tensor.fill([3,4,5,6], 0.5);

var s = reduceMax(x, axis: [0, -1]);
print(s.shape); // [4, 5]

var m = reduceMean(x, axis: [1,2], keepDims: true);
print(m.shape); // [3, 1, 1, 6]
```

### Linear algebra
The set of LA operations is not that big, `loredart` supports batched `matmul` and `matrixTransposition`.
```dart
final a = Tensor.fill([2, 5, 3, 4], 0.1);
final b = Tensor.eye(4, numCols: 7, batchShape: [2, 5]);
print(b.shape); // [2, 5, 4, 7]

final c = matmul(a, b);
print(c.shape); // [2, 5, 3, 7]

final cT = matrixTranspose(c);
print(cT.shape); // [2, 5, 7, 3]
```

### Other ops
Reshaping operations includes `reshape`, `expandDims`, and `squeeze`:
```dart
var x = normal([10, 10]); // shape: [10, 10]
x = expandDims(x, -1);
print(x.shape); // new shape: [10, 10, 1]
```

Also casting with `cast`, concatenation with `concat`, and slicing with `slice`.
Slicing is the only available way to extract sub-tensors:
```dart
var x = Tensor.zeros([5, 8, 13]);

var v = slice(x, [3, 0, 0], [4, 8, 13]);
print(v.shape); // [1, 8, 13]

var y = slice(x, [0, 4, 0], [5, 6, 0]);
print(y.shape); // [5, 2]
```

## `Tensor`s serialization
For some use cases, there are two ways to de- and serialize the `Tensor`: with either json or bytes (binary file).
See `tensorToJson`, `tensorFromJson` and, `tensorToBytes`, `tensorFromBytes`.

## Notes on shape broadcasting
Broadcasting of shapes for the arithmetic and comparison ops is really convenient (for instance, for adding bias in neural network layers). `loredart` consider shapes as broadcastable if at least one of the following criteria is met:
- shapes are equal;
- shapes are compatible (if they are of the same rank, and corresponding dims are either equal or one of them is 1);
- shapes have equal last $k$ dims

See `TensorShape` methods for examples.

## Notes on limitations and future work
There are some limitations, but the most obvious is element extraction (because there is still no need for it), which might appear in the future, even though it won't be as easy to use as in TF.

Also, the next steps are to add support for other `DType`s, and extend the set of LA methods (permutations, matvec, etc.)

---
> ![](assets/GloryForUkraineMini.png) If you would like to support UKRAINE with a donation, visit [**UNITED24**](https://u24.gov.ua/) for more info. Thanks.