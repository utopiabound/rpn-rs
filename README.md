# rpn-rs

RPN-rs is an RPN calculator written in rust using FLTK.

## Goals
* RPN Calculator handling the following:
   * Arbitrary precision rational arithmetic
   * Large precision floating point arithmetic
   * Complex arithmetic
   * Matrixes of the above
 * Simple stack view

## Linux (Fedora) Build Requires
* libstdc++-static
* libpng-devel
* libjpeg-devel
* zlib-devel

## Inspired by
 * [GRPN](https://github.com/utopiabound/grpn)

## Known Issues
* Matrix Inverse not always correct [https://github.com/wiebecommajonas/libmat/issues/4]
