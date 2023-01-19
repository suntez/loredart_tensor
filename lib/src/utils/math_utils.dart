import 'dart:math';

/// Computes hyperbolic sine of [x]
double sinh(double x) => (exp(x) - exp(-x)) / 2;

/// Computes hyperbolic cosine of [x]
double cosh(double x) => (exp(x) + exp(-x)) / 2;

/// Computes hyperbolic tan of [x]
double tanh(double x) => sinh(x) / cosh(x);

/// Computes hyperbolic sec of [x]
double sech(double x) => 1 / cosh(x);