import 'dart:math' show min;

/// The shape of a [Tensor].
/// 
/// Represents number of dimensions and a size for each dimension of a [Tensor].
class TensorShape {
  late final List<int> list;
  late final int rank;
  int get size => list.reduce((d1, d2) => d1 * d2);

  TensorShape(List<int> shape) {
    if (shape.isEmpty || shape.any((dim) => dim <= 0)) {
      throw ArgumentError('shape must be not empty list of positive integers, but received $shape', 'shape');
    }
    list = shape;
    rank = list.length;
  }

  int operator [](int index) {
    if (index >= -rank && index < rank) {
      return index >= 0 ? list[index] : list[rank + index];
    } else {
      throw ArgumentError();
    }
  }

  bool equalWith(TensorShape other) {
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

  bool equalWithLastDims(TensorShape other) {
    final int lastDim = min(other.rank, rank);
    for (int i = 1; i <= lastDim; i += 1) {
      if (this[-i] != other[-i]) {
        return false;
      }
    }
    return true;
  }

  bool broadcastableWith(TensorShape other) => equalWith(other) || compatibleWith(other) || equalWithLastDims(other);

  @override
  String toString() => list.toString();
}