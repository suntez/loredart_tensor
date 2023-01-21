// import 'dart:typed_data';
import 'package:loredart_tensor/loredart_tensor.dart';
import 'package:loredart_tensor/src/ops/comparison_ops.dart';

void main() {
  final x = Tensor.constant([1,2,3,4]);
  final y = Tensor.constant([1,2,2,4]);
  print(equal(x, y));
}