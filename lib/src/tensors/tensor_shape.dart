import 'dart:math' show min;

/// The shape of a [Tensor].
/// 
/// Represents a number of dimensions and a size for each dimension of a [Tensor].
class TensorShape {
  /// The list of dims
  late final List<int> list;

  /// A number of dims in the [shape].
  int get rank => list.length;

  /// A total number of elements.
  int get size => list.reduce((d1, d2) => d1 * d2);

  /// Creates new TensorShape from non-empty [shape].
  /// 
  /// [shape] entries must be positive integers, otherwise will throw an ArgumentError.
  /// 
  /// Example:
  /// ```dart
  /// final shape = TensorShape([13, 8, 5]);
  /// print(shape); // [13, 8, 5]
  /// print(shape.list); // [13, 8, 5]
  /// print(shape.rank); // 3
  /// print(shape.size); // 520
  /// print(shape[0]); // 13
  /// print(shape[-2]); // 8
  /// ```
  TensorShape(List<int> shape) {
    if (shape.isEmpty || shape.any((dim) => dim <= 0)) {
      throw ArgumentError('shape must be non-empty list of positive integers, but received $shape', 'shape');
    }
    list = List.from(shape, growable: false);
  }

  /// The dimension size at the given [index] in the shape.
  /// 
  /// [TensorShape] supports negative indices, so [index] can be from a range `[-rank, rank)`.
  /// Example:
  /// ```dart
  /// final shape = TensorShape([13, 8, 5]);
  /// print(shape); // [13, 8, 5]
  /// print(shape.list); // [13, 8, 5]
  /// print(shape.rank); // 3
  /// print(shape.size); // 520
  /// print(shape[0]); // 13
  /// print(shape[-2]); // 8
  /// ```
  int operator [](int index) {
    if (index >= -rank && index < rank) {
      return index >= 0 ? list[index] : list[rank + index];
    } else {
      throw RangeError.value(index, 'index', 'Index value $index is out of the range [-$rank, $rank)');
    }
  }

  /// Whether this shape is equal to [other].
  /// 
  /// Shapes are considered equal if they are of the same rank and corresponding dims are equal.
  /// 
  /// Example:
  /// ```dart
  /// final shape1 = TensorShape([13, 8, 5]);
  /// 
  /// final shape2 = TensorShape([13, 8, 5]);
  /// shape1.equalTo(shape2); // true
  /// 
  /// final shape3 = TensorShape([13, 8, 4]);
  /// shape1.equalTo(shape3); // false
  /// ```
  bool equalTo(TensorShape other) {
    if (rank == other.rank) {
      for (int i = 0; i < rank; i += 1) {
        if (list[i] != other[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  /// Whether this shape is compatible with [other].
  /// 
  /// Shapes are considered compatible if they are of the same rank, and corresponding dims are either equal or one of them is 1.
  /// 
  /// If shapes are equal, they are compatible as well.
  /// 
  /// Example:
  /// ```dart
  /// final shape1 = TensorShape([13, 8, 5]);
  /// 
  /// final shape2 = TensorShape([13, 8, 1]);
  /// shape1.compatibleWith(shape2); // true
  /// 
  /// final shape3 = TensorShape([13, 1, 1, 1]);
  /// shape1.compatibleWith(shape3); // false
  /// ```
  bool compatibleWith(TensorShape other) {
    if (rank == other.rank) {
      for (int i = 0; i < rank; i += 1) {
        if (list[i] != other[i] && list[i] != 1 && other[i] != 1) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  /// Whether this shape is a subshape of [other] or vice versa.
  /// 
  /// Shapes are considered equalWithLastDims if they have the same last k dims equal.
  /// 
  /// If shapes are equal, they are equalWithLastDims as well.
  /// Example:
  /// ```dart
  /// final shape1 = TensorShape([13, 8, 5]);
  /// 
  /// final shape2 = TensorShape([8, 5]);
  /// shape1.equalWithLastDims(shape2); // true
  /// 
  /// final shape3 = TensorShape([13]);
  /// shape1.equalWithLastDims(shape3); // false
  /// ```
  bool equalWithLastDims(TensorShape other) {
    final int lastDim = min(other.rank, rank);
    for (int i = 1; i <= lastDim; i += 1) {
      if (this[-i] != other[-i]) {
        return false;
      }
    }
    return true;
  }

  /// Whether this shape is broadcastable with [other].
  /// 
  /// Shapes are considered broadcastable if they are either equal or one compatible or have common last k dims.
  /// 
  /// Arithmetic and comparison operations support [Tensor]s with broadcastable shapes.
  /// 
  /// Example:
  /// ```dart
  /// final shape1 = TensorShape([13, 8, 5]);
  /// 
  /// final shape2 = TensorShape([13, 8, 5]);
  /// shape1.broadcastableWith(shape2); // true
  /// 
  /// final shape3 = TensorShape([1, 8, 1]);
  /// shape1.broadcastableWith(shape3); // true
  /// 
  /// final shape4 = TensorShape([5]);
  /// shape1.broadcastableWith(shape4); // true
  /// 
  /// final shape5 = TensorShape([13, 8]);
  /// shape1.broadcastableWith(shape5); // false
  /// 
  /// final shape6 = TensorShape([13, 1, 1, 1]);
  /// shape1.broadcastableWith(shape6); // false
  /// ```
  bool broadcastableWith(TensorShape other) => equalTo(other) || compatibleWith(other) || equalWithLastDims(other);

  @override
  String toString() => list.toString();
}