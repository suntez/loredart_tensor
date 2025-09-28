/// Returns the shape of the nested lists of [values].
///
/// All nested lists must be of the same length, otherwise will throw an ArgumentError.
List<int> extractDimsFromNestedValues(List<dynamic> values) {
  List<int> shape = [];
  while (true) {
    shape.add(values.length);
    if (values[0] is List) {
      try {
        if (values.any((element) => element.length != values[0].length)) {
          throw ArgumentError('All lists must have equal number of elements', 'values');
        }
        values = values[0];
      } catch (e) {
        rethrow;
      }
    } else {
      break;
    }
  }
  return shape;
}

/// Recursively flatten nested [values].
///
/// The elements of the flattened list must be of the same type, otherwise will throw an ArgumentError.
List flattenList(List values) {
  if (values.every((element) => element is List)) {
    final flattenLists = [for (final value in values) ...flattenList(value)];
    return flattenLists;
  } else {
    if (values.any((element) => element.runtimeType != values[0].runtimeType)) {
      throw ArgumentError('All entities of values must have the same Type', 'values');
    }
    return values;
  }
}
