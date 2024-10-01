# rpn-rs

RPN-rs is an RPN calculator written in rust using `FLTK`.

Also can be run as command line or as a graphical text user interface.

[![Code Coverage](https://codecov.io/gh/utopiabound/rpn-rs/graph/badge.svg?token=4HQJXC0TNK)](https://codecov.io/gh/utopiabound/rpn-rs)
![Build Status](https://github.com/utopiabound/rpn-rs/workflows/RPN%20Push/badge.svg)

## Features

* Number - Scalar (via [`rug`](latest/rug/struct.Integer.html) crate)
   * Arbitrary precision rational numbers ([`GMP`](https://gmplib.org/))
   * High-precision floating-point ([`MPFR`](https://www.mpfr.org/))
   * High-precision complex numbers ([`MPC`](https://www.multiprecision.org/mpc/))
* Numbers - Matrix (via [`libmat`](https://github.com/wiebecommajonas/libmat))
   * Matrices of any of above scalars
   * Correct interaction between scalars and Matrices
* Numbers - Tuples
   * Vector of any scalar value
   * Allow for statistical functions on groups of data
   * Allow for stack functions on tuples (push/pop)
* User-Interface
   * `GUI` - Graphical User Interface
   * `TUI` - Full terminal user interface
   * `CLI` - Basic interactive command line

## Goals

* RPN Calculator
* Arbitrary precision
* Matrices
* Simple stack view

## Linux (Fedora) Build Requires

* `libstdc++-static`
* `libpng-devel`
* `libjpeg-devel`
* `zlib-devel`

## Inspired by

* [`GRPN`](https://github.com/utopiabound/grpn)
* HP RPN Calculators (e.g. HP48G)
