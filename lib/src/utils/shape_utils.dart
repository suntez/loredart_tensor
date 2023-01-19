List<int> extractDimsFromNestedValues(List<dynamic> values) {
  List<int> shape = [];
  while (true) {
    shape.add(values.length);
    if (values[0] is List) {
      try {
        if (values.any((element) => element.length != values[0].length)) {
          throw ArgumentError('All lists must have equal number of elements');
        }
        values = values[0];
      } catch(e) {
        rethrow;
      }
    } else {
      break;
    }
  }
  return shape;
}

List flattenList(List list) {
  if (list.every((element) => element is List)) {
    final flattenLists = [for (final values in list) ...flattenList(values)];
    return flattenLists;
  } else {
    if (list.any((element) => element.runtimeType != list[0].runtimeType)) {
      throw ArgumentError('All entities must have the same Type');
    }
    return list;
  }
}