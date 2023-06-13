# rpn-rs

RPN-rs is an RPN calculator written in rust using FLTK.

Also can be run as CLI or as a TUI.

## Features

* Number - Scaler (via [rug](latest/rug/struct.Integer.html) crate)
   * Arbitrary precision rational numbers ([GMP](https://gmplib.org/))
   * High-precision floating-point ([MPFR](https://www.mpfr.org/))
   * Complex numbers ([MPC](https://www.multiprecision.org/mpc/))
* Numbers - Matrix (via [libmat](https://github.com/wiebecommajonas/libmat))
   * Matricies of any of above Scalers
   * Correct interaction between Scalers and Matricies
* UI
   * GUI - Graphical User Interface
   * TUI - Full terminal user interface
   * CLI - Basic interacive CLI

## Goals

* RPN Calculator
* Arbitrary precision
* Matrixes
* Simple stack view

## Linux (Fedora) Build Requires

* libstdc++-static
* libpng-devel
* libjpeg-devel
* zlib-devel

## Inspired by

* [GRPN](https://github.com/utopiabound/grpn)
* HP RPN Calculators (e.g. HP48G+)
