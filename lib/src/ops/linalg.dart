import '../tensors/num_tensor.dart';
import '../tensors/tensor.dart';
import '../utils/dtype_utils.dart';

/// Multiplies matrix [a] by matrix [b].
/// 
/// The input tensors must, following any transpositions given in [transposeA] or [transposeB],
/// be of rank >= 2 where the inner 2 dimensions specify valid matrix multiplication dimensions,
/// and any further outer dimensions specify matching batch size, or otherwise will throw an ArgumentError.
/// 
/// Returns a (batched) matrix with the same [DType] as [a] and [b].
/// 
/// Example of simple matrix multiplication (with 2D tensors):
/// ```dart
/// Tensor a = Tensor.ones([3, 4]);
/// Tensor b = Tensor.diag([1.0, 2.0, 3.0, 4.0], numCols: 5); // shape is [4,5]
/// final c = matmul(a, b); // output shape is [3,5]
/// // <Tensor(shape: [3, 5], values:
/// //  [[1.0, 2.0, 3.0, 4.0, 0.0]
/// //   [1.0, 2.0, 3.0, 4.0, 0.0]
/// //   [1.0, 2.0, 3.0, 4.0, 0.0]], dType: float32)>
/// ```
/// `Note`: For transposition, it's better to use [transposeA] or [transposeB] parameters rather than the `matrixTranspose` operation.
/// ```dart
/// final c2 = matmul(b, a, transposeA: true, transposeB: true); // output shape is [3,5]
/// ```
/// 
/// Example of batch matrix multiplication:
/// ```dart
/// Tensor a = Tensor.ones([2, 2, 3, 4]);
/// Tensor b = Tensor.eye(4, batchShape: [2,2]); // shape is [2, 2, 4, 4]
/// final c = matmul(a, b);
/// print(c.shape); // [2, 2, 3, 4]
/// ```
Tensor matmul(Tensor a, Tensor b, {bool transposeA = false, bool transposeB = false}) {
  if (a.rank != b.rank) {
    throw ArgumentError('Tensors a and b must be of the same rank, but received a.rank: ${a.rank} and b.rank: ${b.rank}');
  }
  
  for (int i = 0; i < a.rank-2; i += 1) {
    if (a.shape[i] != b.shape[i]) {
      throw ArgumentError('Tensors a and b must have same batch shape, but received a.shape: ${a.shape} and b.shape: ${b.shape}');
    }
  }
  
  if ((a is NumericTensor) && (b is NumericTensor)) {
    DType dType = dTypeDecision(a.dType, b.dType);

    final aM = transposeA ? a.shape[-1] : a.shape[-2];
    final aN = transposeA ? a.shape[-2] : a.shape[-1]; // equal to bN
    final bN = transposeB ? b.shape[-1] : b.shape[-2]; // equal to aN
    final bL = transposeB ? b.shape[-2] : b.shape[-1];

    if (aN != bN) {
      throw ArgumentError('Last dims of a and b should meet requirements of matrix multiplication, but $aN != $bN');
    }

    final int combinedBatchSize = a.shape.size ~/ aM ~/ aN;

    List<int> shape = [
      ...a.shape.list.sublist(0, a.rank - 2),
      aM,
      bL
    ];

    List buffer = emptyBuffer(dType, combinedBatchSize * aM * bL);
    for (int batch = 0; batch < combinedBatchSize; batch += 1) {
      for (int i = 0; i < aM; i += 1) {
        for (int j = 0; j < bL; j += 1) {
          num dot = dType.isInt ? 0 : 0.0;
          for (int k = 0; k < aN; k += 1) {
            dot += a.buffer[batch * aM * aN + (!transposeA ? (i * aN + k) : (k * aM + i))] *
                b.buffer[batch * bN * bL + (!transposeB ? (k * bL + j) : (k + j * bN))];
          }
          buffer[batch * aM * bL + i * bL + j] = dot;
        }
      }
    }
    return Tensor.fromTypedDataList(buffer, shape, dType: dType);
  } else {
    throw ArgumentError('Tensors a and b must be NumericTensors of the same DType, but received ${a.dType} and ${b.dType}');
  }
}

/// Transposes last two dimensions of [x].
/// 
/// If [x.shape] is `[b1, b2, ..., n, m]`, then the output shape is `[b1, b2, ..., m, n]`.
/// 
/// Returns [Tensor] of the same [DType] as [x].
/// 
/// Example:
/// ```dart
/// Tensor x = Tensor.fill([2,3,4,5], 2.71);
/// final xT = matrixTranspose(x);
/// print(xT.shape); // [2, 3, 5, 4]
/// ```
/// 
/// `Note`: For transpositions in `matmul`, it's better to use [transposeA] or [transposeB] parameters, rather than the `matrixTranspose` operation.
/// ```dart
/// Tensor c1 = matmul(a, b, transposeA: true); // Preferable choice, more efficient 
/// Tensor c2 = matmul(matrixTranspose(a), b); // Non-preferable choice, works slower
/// ```
Tensor matrixTranspose(Tensor x) {
  if (x is NumericTensor && x.rank >= 2) {
    List buffer = emptyBuffer(x.dType, x.shape.size);
    final matrixSize = x.shape[-1] * x.shape[-2];
    for (int b = 0; b < x.shape.size ~/ matrixSize; b += 1) {
      for (int i = 0; i < x.shape[-2]; i += 1) {
        for (int j = 0; j < x.shape[-1]; j += 1) {
          buffer[b*matrixSize + j * x.shape[-2] + i] = x.buffer[b*matrixSize + i * x.shape[-1] + j];
        }
      }
    }
    List<int> shape = [...x.shape.list.sublist(0, x.rank-2), x.shape[-1], x.shape[-2]];
    return Tensor.fromTypedDataList(buffer, shape, dType: x.dType);
  } else {
    throw ArgumentError('Expected NumericTensors with rank >= 2, but received ${x.runtimeType} of the rank: ${x.rank}', 'x');
  }
}

//TODO: matvec, det, transpose with permutations