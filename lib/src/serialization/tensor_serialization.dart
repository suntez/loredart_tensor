import 'dart:convert';
import 'dart:typed_data';

import '../tensors/tensor.dart';
import '../tensors/num_tensor.dart';
import '../utils/dtype_utils.dart';

// enum SerializationType {json, loreTensor, bin}

/// Converts [tensor] into a json-encodable Map.
/// 
/// Designed to store [Tensor]s as json files.
/// Usually, the bytes representation by [tensorToBytes] is less memory-consuming,
/// but it might depend on the actual values from Tensor.
/// 
/// Example:
/// ```dart
/// final x = Tensor.constant(<double>[1,2,3,4]);
/// Map mapData = tensorToJson(x);
/// 
/// someFile.writeAsString(json.encode(mapData));
/// ```
Map<String, dynamic> tensorToJson(Tensor tensor) {
  if (tensor is NumericTensor) {
    return {'dType': tensor.dType.name, 'shape': tensor.shape.list, 'buffer': tensor.buffer};
  } else {
    throw ArgumentError('Only NumericTensor(s) are json serializable, but received ${tensor.runtimeType}', 'tensor');
  }
}

/// Constructs a [Tensor] from a Map [data] structure.
/// 
/// Designed to deserialize [Tensor] from json file.
/// 
/// Example:
/// ```dart
/// String jsonData = someFile.readAsStringSync();
/// Tensor x = tensorFromJson(json.decode(jsonData)); 
/// ```
Tensor tensorFromJson(Map<String, dynamic> data) {
  if (data.containsKey('buffer') && data.containsKey('shape') && data.containsKey('dType')) {
  return Tensor.constant(data['buffer'], shape: (data['shape'] as List).cast<int>(), dType: DType.values.byName(data['dType']));
  } else {
    throw ArgumentError("Incorrect json data, expected 'buffer', 'shape' and 'dType' keys, but received ${data.keys}");
  }
}

/// Converts [tensor] into a sequence of raw bytes.
/// 
/// Designed to store [Tensor]s as binary files (usually with the `.loretensor` extension, but it doesn't matter).
/// 
/// Usually, the bytes representation by [tensorToBytes] is less memory-consuming than json one
/// because the tensor values are stored as raw bytes, not as Strings,
/// but it might depend on the actual values from the Tensor.
/// 
/// Example:
/// ```dart
/// File someFile = File('x.loretensor'); // or 'x.bin'
/// 
/// final x = Tensor.constant(<double>[1,2,3,4]);
/// var bytes = tensorToBytes(x);
///
/// someFile.writeAsBytes(bytes);
/// ```
List<int> tensorToBytes(Tensor tensor) {
  if (tensor is NumericTensor) {
    Uint8List head = ascii.encode(json.encode({'dType': tensor.dType.name, 'shape': tensor.shape.list}) + '\n');
    return head + (tensor.buffer as TypedData).buffer.asUint8List();
  } else {
    throw ArgumentError('Only NumericTensor(s) are bytes serializable, but received ${tensor.runtimeType}', 'tensor');
  }
}

/// Constructs a [Tensor] from [bytes].
/// 
/// Designed to deserialize [Tensor] from binary data, usually stored as `.loretensor` or `.bin` files.
/// 
/// Example:
/// ```dart
/// File someFile = File('filename.loretensor');
/// final bytes = someFile.readAsBytesSync();
/// Tensor x = tensorFromBytes(bytes); 
/// ```
Tensor tensorFromBytes(List<int> bytes) {
  int divide = bytes.indexOf(0x0A);
  if (divide == -1) {
    throw ArgumentError('Incorrect bytes', 'bytes');
  }
  Map<String, dynamic> head = json.decode(ascii.decode(bytes.sublist(0, divide)));
  if (head.containsKey('shape') && head.containsKey('dType')) {
    DType dType = DType.values.byName(head['dType']);
    return Tensor.fromTypedDataList(
      convertBufferToTypedDataList(Uint8List.fromList(bytes.sublist(divide+1)).buffer, dType),
      (head['shape'] as List).cast<int>(),
      dType: dType
    );
  } else {
    throw ArgumentError('Incorrect bytes', 'bytes');
  }
}

