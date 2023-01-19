import 'dart:math' as math;
import '../tensors/tensor.dart';
import '../utils/dtype_utils.dart';

/// Fills buffer for batched [numRows] by [numCols] identity matrix.
List eyeBuffer(int numRows, int numCols, List<int> batchShape, DType dType) {
  if (numCols <= 0 || numRows <= 0) {
    throw ArgumentError('Number of columns and rows must be > 0, but received numRows: $numRows, numCols: $numCols');
  }
  final int len = (batchShape + [numRows, numCols]).reduce((e1, e2) => e1 * e2);
  List buffer = emptyBuffer(dType, len);
  final int batchSize = batchShape.isEmpty ? 0 : len ~/ numRows ~/ numCols - 1;
  for (int b = 0; b <= batchSize; b += 1) {
    for (int i = 0; i < math.min(numRows, numCols); i += 1) {
        buffer[b * numRows * numCols + i * numCols + i] = dType.isInt ? 1 : 1.0;
    }
  }
  return buffer;
}

/// Constructs diag [Tensor] from elements of [diagonal], with given [offset].
/// 
/// [numRows] and [numCols] might be used to change shape of matrix.
Tensor createDiagTensor(List<dynamic> diagonal, DType dType, int offset, int? numRows, int? numCols) {
  if (diagonal[0] is num) {
    if (diagonal.any((element) => element.runtimeType != diagonal[0].runtimeType)) {
      throw ArgumentError('All entities must be of the same type', 'diagonal');
    }

    numRows ??= diagonal.length + offset.abs();
    numCols ??= diagonal.length + offset.abs();
    if (numCols < diagonal.length || numRows < diagonal.length) {
      throw ArgumentError('numRows and numCols must be equal or greater then length of the diagonal elements with offset, but received numCols: $numCols, numRows: $numRows',);
    }
    final int length = numCols * numRows;
    dType = dTypeAndNumDecision(dType, diagonal[0].runtimeType, true);
    List buffer = emptyBuffer(dType, length);
    for (int i = 0; i < diagonal.length; i += 1) {
      buffer[i * numCols + i + (offset >= 0 ? offset : (-offset)*numCols)] = dType.isInt ? (diagonal[i] as num).toInt() : (diagonal[i] as num).toDouble();
    }
    return Tensor.fromBuffer(buffer, [numRows, numCols], dType: dType);
  }
  else if (diagonal[0] is List<num>) {

    final int lengthOfDiag = diagonal[0].length;
    if (diagonal.any((element) => element.length != lengthOfDiag)) {
      throw ArgumentError('Length of all diagonal elements must be equal', 'diagonal');
    }
    if (diagonal.any((list) => list.any((element) => element.runtimeType != diagonal[0][0].runtimeType))) {
      throw ArgumentError('All entities must be of the same type', 'diagonal');
    }
    dType = dTypeAndNumDecision(dType, diagonal[0][0].runtimeType, true);

    numRows ??= lengthOfDiag + offset.abs();
    numCols ??= lengthOfDiag + offset.abs();

    if (numCols < lengthOfDiag || numRows < lengthOfDiag) {
      throw ArgumentError('numRows and numCols must be >= length of the diagonal(s) elements with offset, but received numCols: $numCols, numRows: $numRows');
    }

    final int length = diagonal.length * numCols * numRows;
    List buffer = emptyBuffer(dType, length);
    for (int b = 0; b < diagonal.length; b += 1) {
      for (int i = 0; i < diagonal[0].length; i += 1) {
        buffer[b*numRows*numCols + i * numCols + i + (offset >= 0 ? offset : (-offset)*numCols)] = dType.isInt ? (diagonal[b][i] as num).toInt() : (diagonal[b][i] as num).toDouble();
      }
    }
    return Tensor.fromBuffer(buffer, [diagonal.length, numRows, numCols], dType: dType);
  } else {
    throw ArgumentError('Expected List<num> or List<List<num>> as diagonal, but received ${diagonal.runtimeType}', 'diagonal');
  }
}