import 'dart:math' show Random, log, sin, cos, sqrt, pi;

List<double> generateNormallyDistList(int length, num mean, num std, {int? seed,}) {
  final Random randU1 = Random(seed);
  final Random randU2 = Random(seed);
  List<double> generateNormPair() {
    final u1 = randU1.nextDouble();
    final u2 = randU2.nextDouble();
    final radius = sqrt(-2 * log(u1));
    final theta = 2*pi*u2;
    return [radius * cos(theta), radius * sin(theta)];
  }
  List<double> normallyDist01 = [for (int i = 0; i < length~/2; i += 1) ...generateNormPair()];
  if (length.isOdd) {
    normallyDist01.add(generateNormPair()[0]);
  }
  return List.generate(length, (i) => normallyDist01[i] * std + mean);
}


List<double> generateTruncatedNormallyDistList(int length, num mean, num std, {int? seed,}) {
  final Random randU1 = Random(seed);
  final Random randU2 = Random(seed);
  List<double> generateTruncatedNormPair([int iter = 0, bool onlySin = false]) {
    final u1 = randU1.nextDouble();
    final u2 = randU2.nextDouble();
    final radius = sqrt(-2 * log(u1));
    final theta = 2*pi*u2;
    double v1 = radius * cos(theta);
    double v2 = radius * sin(theta);
    if (v1 > 2 && iter < 100 && !onlySin) {
      v1 = generateTruncatedNormPair(iter+1, false)[0];
    }
    if (v2 > 2 && iter < 100 && onlySin) {
      v2 = generateTruncatedNormPair(iter+1, true)[1];
    }
    return [v1, v2];
  }
  List<double> normallyDist01 = [for (int i = 0; i < length~/2; i += 1) ...generateTruncatedNormPair()];
  if (length.isOdd) {
    normallyDist01.add(generateTruncatedNormPair()[0]);
  }
  return List.generate(length, (i) => normallyDist01[i] * std + mean);
}