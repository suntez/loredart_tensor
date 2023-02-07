import 'dart:math' as math;

import '../tensors/num_tensor.dart';
import '../utils/math_utils.dart' as hyperbolic;
import '../tensors/tensor.dart';
import '../utils/dtype_utils.dart';

/// Applies [func] on [x] element-wise.
/// 
/// Returns [Tensor] with either given [dType] or [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor apply(Tensor x, num Function(num) func, {DType? dType}) {
    if (x is NumericTensor) {
    dType ??= (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = func(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes absolute value of [x] element-wise.
/// 
/// Returns [Tensor] with same [DType] as [x].
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor abs(Tensor x) {
  if (x is NumericTensor) {
      List buffer = emptyBuffer(x.dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = (x.buffer[i] as num).abs();
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: x.dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes exponent of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor exp(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.exp(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes e^[x]-1 element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor expm1(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.exp(x.buffer[i])-1;
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes sine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor sin(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.sin(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes cosine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor cos(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.cos(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes tan of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor tan(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.tan(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes arc cosine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor acos(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.acos(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes arc sine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor asin(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.asin(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes arc tan of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor atan(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.atan(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes hyperbolic sine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor sinh(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = hyperbolic.sinh(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes hyperbolic cosine of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor cosh(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = hyperbolic.cosh(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes hyperbolic tan of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor tanh(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = hyperbolic.tanh(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes hyperbolic sec of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor sech(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = hyperbolic.sech(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes the natural logarithm of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor log(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.log(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes the natural logarithm of (1+[x]) element-wise.
/// 
/// `y_i = log(1+x_i)`
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor log1p(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.log(x.buffer[i] + 1.0);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes the binary logarithm of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor log2(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.log(x.buffer[i]) / math.log2e;
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes the positive square of [x] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor sqrt(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.sqrt(x.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes square of [x] element-wise.
/// 
/// Returns [Tensor] with same [DType] as [x].
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor square(Tensor x) {
  if (x is NumericTensor) {
      List buffer = emptyBuffer(x.dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] * x.buffer[i];
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: x.dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes [x] to the power of [exponent] element-wise.
/// 
/// [exponent] could be a [Tensor] of the same shape as [x] or be a [num].
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor] or if [exponent.shape] != [x.shape] (if [exponent] is [Tensor]).
Tensor pow(Tensor x, Object exponent) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      if (exponent is num) {
        for (int i = 0; i < x.shape.size; i += 1) {
          buffer[i] = math.pow(x.buffer[i], exponent);
        }
      } else if (exponent is NumericTensor && x.shape.equalTo(exponent.shape)) {
        for (int i = 0; i < x.shape.size; i += 1) {
          buffer[i] = math.pow(x.buffer[i], exponent.buffer[i]);
        }
      } else {
        throw ArgumentError('Expected NumericTensor with same shape or number for exponent arg, but received ${exponent.runtimeType}');
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes an element-wise indication of the sign of [x].
/// 
/// Returns [Tensor] with same [DType] as [x].
Tensor sign(Tensor x) {
  if (x is NumericTensor) {
      List buffer = emptyBuffer(x.dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        if (x.dType.isInt) {
          buffer[i] = x.buffer[i] > 0 ? 1 : x.buffer[i] == 0 ? 0 : -1;
        } else {
          buffer[i] = x.buffer[i] > 0.0 ? 1.0 : x.buffer[i] == 0.0 ? 0.0 : -1.0;
        }
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: x.dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes sigmoid of [x] element-wise.
/// 
/// `sigmoid(x) = 1/(1+exp(-x))`
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor sigmoid(Tensor x) {
    if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = 1/(1 + math.exp(-x.buffer[i]));
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes softplus of [x] element-wise.
/// 
/// `softplus(x) = log(1+exp(x))`
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor softplus(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = math.log(math.exp(x.buffer[i])+1);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes softminus of [x] element-wise.
/// 
/// `softminus(x) = x - softplus(x)`
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor softminus(Tensor x) {
  if (x is NumericTensor) {
    final dType = (x.dType == DType.float64) ? x.dType : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] -  math.log(math.exp(x.buffer[i])+1);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes ([x]-[y])^2 element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [y.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// [x] and [y] expected to be [NumericTensor]s of the same [DType] with equal [TensorShapes], otherwise will throw an ArgumentError.
Tensor squareDifference(Tensor x, Tensor y) {
  if (x is NumericTensor && y is NumericTensor && x.dType == y.dType) {
      if (x.shape.equalTo(y.shape)) {
      final dType = (x.dType == DType.float64 && y.dType == DType.float64) ? DType.float64 : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = (x.buffer[i] - y.buffer[i])*(x.buffer[i] - y.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
    } else {
      throw ArgumentError('Operands must be of the same shape, but received x.shape: ${x.shape}, y.shape: ${y.shape}');
    }
  } else {
    throw ArgumentError(
        'Expected NumericTensors of the same DType, but received x: ${x.runtimeType} and y: ${y.runtimeType}');
  }
}

/// Computes [x]*log([y]) element-wise.
/// 
/// For x_i == 0 safely returns 0, regardless to the value of y_i.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [y.dType] == [DType.float64] or [DType.float32] otherwise.
/// 
/// [x] and [y] expected to be [NumericTensor]s of the same [DType] with equal [TensorShapes], otherwise will throw an ArgumentError.
Tensor xlogy(Tensor x, Tensor y) {
  if (x is NumericTensor && y is NumericTensor && x.dType == y.dType) {
      if (x.shape.equalTo(y.shape)) {
      final dType = (x.dType == DType.float64 && y.dType == DType.float64) ? DType.float64 : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] == 0 ? (dType.isInt ? 0 : 0.0) : x.buffer[i] * math.log(y.buffer[i]);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
    } else {
      throw ArgumentError('Operands must be of the same shape, but received x.shape: ${x.shape}, y.shape: ${y.shape}');
    }
  } else {
    throw ArgumentError(
        'Expected NumericTensors of the same DType, but received x: ${x.runtimeType} and y: ${y.runtimeType}');
  }
}

/// Computes [x]*log([y]+1) element-wise.
/// 
/// For x_i == 0 safely returns 0, regardless to the value of y_i.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [y.dType] == [DType.float64] or [DType.float32] otherwise.
///
/// [x] and [y] expected to be [NumericTensor]s of the same [DType] with equal [TensorShape]s, otherwise will throw an ArgumentError.
Tensor xlog1py(Tensor x, Tensor y) {
  if (x is NumericTensor && y is NumericTensor && x.dType == y.dType) {
      if (x.shape.equalTo(y.shape)) {
      final dType = (x.dType == DType.float64 && y.dType == DType.float64) ? DType.float64 : DType.float32;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] == 0 ? 0 : x.buffer[i] * math.log(y.buffer[i]+1);
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
    } else {
      throw ArgumentError('Operands must be of the same shape, but received x.shape: ${x.shape}, y.shape: ${y.shape}');
    }
  } else {
    throw ArgumentError(
        'Expected NumericTensors of the same DType, but received x: ${x.runtimeType} and y: ${y.runtimeType}');
  }
}

/// Computes max of [x] and [y] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [y.dType] == [DType.float64] or [DType.float32] otherwise.
///
/// [x] and [y] expected to be [NumericTensor]s of the same [DType] with equal [TensorShapes], otherwise will throw an ArgumentError.
Tensor max(Tensor x, Tensor y) {
  if (x is NumericTensor && y is NumericTensor && x.dType == y.dType) {
      if (x.shape.equalTo(y.shape)) {
      final dType = x.dType;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] >= y.buffer[i] ? x.buffer[i] : y.buffer[i];
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
    } else {
      throw ArgumentError('Operands must be of the same shape, but received x.shape: ${x.shape}, y.shape: ${y.shape}');
    }
  } else {
    throw ArgumentError(
        'Expected NumericTensors of the same DType, but received x: ${x.runtimeType} and y: ${y.runtimeType}');
  }
}

/// Computes min of [x] and [y] element-wise.
/// 
/// Returns [Tensor] with either [DType.float64] if [x.dType] == [y.dType] == [DType.float64] or [DType.float32] otherwise.
///
/// [x] and [y] expected to be [NumericTensor]s of the same [DType] with equal [TensorShapes], otherwise will throw an ArgumentError.
Tensor min(Tensor x, Tensor y) {
  if (x is NumericTensor && y is NumericTensor && x.dType == y.dType) {
      if (x.shape.equalTo(y.shape)) {
      final dType = x.dType;
      List buffer = emptyBuffer(dType, x.shape.size);
      for (int i = 0; i < x.shape.size; i += 1) {
        buffer[i] = x.buffer[i] <= y.buffer[i] ? x.buffer[i] : y.buffer[i];
      }
      return Tensor.fromTypedDataList(buffer, x.shape.list, dType: dType);
    } else {
      throw ArgumentError('Operands must be of the same shape, but received x.shape: ${x.shape}, y.shape: ${y.shape}');
    }
  } else {
    throw ArgumentError(
        'Expected NumericTensors of the same DType, but received x: ${x.runtimeType} and y: ${y.runtimeType}');
  }
}

/// Computes max of [x].
/// 
/// Returns single-element [Tensor] with same [DType] as [x].
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor maximum(Tensor x) {
  if (x is NumericTensor) {
    num maximumNum = x.buffer[0];
    for (int i = 0; i < x.shape.size; i += 1) {
      maximumNum = x.buffer[i] > maximumNum ? x.buffer[i] : maximumNum;
    }
    return Tensor.constant([maximumNum], dType: x.dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes min of [x].
/// 
/// Returns single-element [Tensor] with same [DType] as [x].
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor minimum(Tensor x) {
  if (x is NumericTensor) {
    num minimumNum = x.buffer[0];
    for (int i = 0; i < x.shape.size; i += 1) {
      minimumNum = x.buffer[i] < minimumNum ? x.buffer[i] : minimumNum;
    }
    return Tensor.constant([minimumNum], dType: x.dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes mean of [x].
/// 
/// If [dType] isn't specified, then it's inherited from [x].
/// Returns single-element [Tensor].
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor mean(Tensor x, {DType? dType}) {
  if (x is NumericTensor) {
    dType ??= x.dType;
    num sum = dType.isInt ? 0 : 0.0;
    for (int i = 0; i < x.shape.size; i += 1) {
      sum += x.buffer[i];
    }
    return Tensor.constant([sum / x.shape.size], dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}

/// Computes variance of [x].
/// 
/// If [dType] isn't specified, then it's inherited from [x].
/// Returns single-element [Tensor].
/// 
/// Throws an ArgumentError if [x] is not [NumericTensor].
Tensor variance(Tensor x, {DType? dType}) {
  if (x is NumericTensor) {
    dType ??= x.dType;
    num sum = dType.isInt ? 0 : 0.0;
    num squareSum = dType.isInt ? 0 : 0.0;
    for (int i = 0; i < x.shape.size; i += 1) {
      sum += x.buffer[i];
      squareSum += x.buffer[i]*x.buffer[i];
    }
    return Tensor.constant([(squareSum - (sum*sum /  x.shape.size)) / x.shape.size], dType: dType);
  } else {
    throw ArgumentError(
        'Expected NumericTensor, but received ${x.runtimeType}', 'x');
  }
}