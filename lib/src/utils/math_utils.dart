import 'dart:math';

/// Computes hyperbolic sine of [x]
double sinh(num x) => (exp(x) - exp(-x)) / 2;

/// Computes hyperbolic cosine of [x]
double cosh(num x) => (exp(x) + exp(-x)) / 2;

/// Computes hyperbolic tan of [x]
double tanh(num x) => sinh(x) / cosh(x);

/// Computes hyperbolic sec of [x]
double sech(num x) => 1 / cosh(x);