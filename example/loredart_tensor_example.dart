import 'dart:io';

import 'package:loredart_tensor/loredart_tensor.dart';

/// Example usage of the [Tensor]s  
void main(List<String> args) {
  final x = Tensor.diag([1,2,3,4,5,6,7,8], offset: -1, numRows: 10);

  // Tensor properties
  print(x.dType);
  print('shape: ${x.shape}');
  print('rank: ${x.rank}');
  print('size: ${x.shape.size}');

  // Tensor values
  print(x);

  // Operations with Tensors
  final w = Tensor.eye(13, numCols: 9);
  final bias = Tensor.ones([10]);

  var res = softplus(matmul(w, x, transposeB: true) + bias);
  print('Resulting shape: ${res.shape}');

  // Normalizing the Tensor
  final m = reduceMean(res, axis: [-1], keepDims: true);
  final v = reduceVariance(res, axis: [-1], keepDims: true);
  res = (res - m) / sqrt(v + 1e-8);

  // reshape and concat with other
  final y = Tensor.fill(res.shape.list + [5], 0.5);
  res = expandDims(res, -1);

  res = concat([res, y], axis: -1);

  // save
  List<int> bytes = tensorToBytes(res);
  File file = File('res.loretensor')..createSync();
  file.writeAsBytesSync(bytes);
}