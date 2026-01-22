/* RPN-rs (c) 2026 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use itertools::{EitherOrBoth, Itertools as _};
use libmat::mat::Matrix;
use num_traits::{Inv, Num, One, Signed, Zero};
use regex::Regex;
use rug::{
    Complete, Complex, Float, Integer, Rational,
    float::{Constant, Round, Special},
    ops::{DivAssignRound, DivFromRound, Pow, RemRounding},
};
use std::{
    cmp::{Ordering, max, min},
    collections::VecDeque,
    ops,
};

const FLOAT_PRECISION: u32 = 256;

#[derive(Debug, Clone)]
pub(crate) enum Scalar {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Scalar(Scalar),
    Tuple(VecDeque<Scalar>),
    Matrix(Matrix<Scalar>),
}

impl From<Matrix<Scalar>> for Value {
    fn from(x: Matrix<Scalar>) -> Self {
        Value::Matrix(x)
    }
}

impl From<Vec<Scalar>> for Value {
    fn from(x: Vec<Scalar>) -> Self {
        Value::Tuple(x.into_iter().collect())
    }
}

impl From<VecDeque<Scalar>> for Value {
    fn from(x: VecDeque<Scalar>) -> Self {
        Value::Tuple(x)
    }
}

impl<T: Into<Scalar>> From<T> for Value {
    fn from(x: T) -> Self {
        Value::Scalar(x.into())
    }
}

trait RpnMatrixExt {
    fn is_diagonal(&self) -> bool;
    fn try_exp(&self) -> Result<Self, String>
    where
        Self: Sized;
}

impl RpnMatrixExt for Matrix<Scalar> {
    fn is_diagonal(&self) -> bool {
        if self.is_square() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    if i != j && self[i][j] != Scalar::zero() {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }
    fn try_exp(&self) -> Result<Self, String> {
        if !self.is_square() {
            Err("Matrix is not square, cannot compute exp(M)".to_string())
        } else if self.is_diagonal() {
            // https://en.wikipedia.org/wiki/Matrix_exponential#Diagonalizable_case
            let mut m = Matrix::one(self.rows()).map_err(|e| e.to_string())?;
            for i in 0..self.rows() {
                m[i][i] = self[i][i].clone().exp();
            }
            Ok(m)
        } else if (self.clone() * self.clone()).as_ref() == Ok(self) {
            // https://en.wikipedia.org/wiki/Matrix_exponential#Projection_case
            // I + (e - 1)*M (where M*M == M)
            let ident = Self::one(self.rows()).map_err(|e| e.to_string())?;
            (ident + (self.clone() * (Scalar::one().exp() - Scalar::one())))
                .map_err(|e| e.to_string())
        } else {
            // c.f. impl Pow<Value> for Value
            // @@TODO Genral Diagonalizable: https://en.wikipedia.org/wiki/Matrix_exponential#Diagonalizable_case
            Err("NYI: non-trivial exp(M)".to_string())
        }
    }
}

trait RpnIntegerExt {
    fn factorial(&self) -> Result<Self, String>
    where
        Self: Sized + One,
    {
        self.partial_factorial(Self::one())
    }
    fn partial_factorial(&self, m: Self) -> Result<Self, String>
    where
        Self: Sized;
}

impl RpnIntegerExt for Integer {
    fn partial_factorial(&self, m: Self) -> Result<Self, String> {
        if self < &m {
            Err("partial factorial m must be smaller than n".to_string())
        } else if self.is_negative() {
            Err("n must not be negative for n!".to_string())
        } else {
            let mut val = Integer::one();
            let mut i = m;
            while i <= *self {
                val *= i.clone();
                i += 1;
            }
            Ok(val)
        }
    }
}

trait RpnToStringScalar {
    fn to_string_scalar(&self, radix: Radix) -> String {
        self.to_string_scalar_len(radix, None)
    }
    fn digits(&self, radix: Radix) -> usize {
        self.to_string_scalar_len(radix, None).len()
    }
    fn to_string_width(&self, radix: Radix, width: Option<usize>) -> String {
        self.to_string_scalar_len(radix, width)
    }
    fn to_string_scalar_len(&self, radix: Radix, digits: Option<usize>) -> String;
}

impl RpnToStringScalar for Float {
    fn to_string_scalar_len(&self, radix: Radix, digits: Option<usize>) -> String {
        let (sign, s, exp) = self.to_sign_string_exp(radix.into(), digits);
        let len = s.len() as i32;

        let s = if let Some(exp) = exp {
            if exp > 0 {
                if exp >= len {
                    let s = if let Some(digits) = digits
                        && exp as usize > digits
                    {
                        let (_, s, _) =
                            self.to_sign_string_exp(radix.into(), Some((exp - len) as usize));
                        s
                    } else {
                        s
                    };
                    s + &"0".repeat((exp - len) as usize)
                } else {
                    let mut v = s;
                    v.insert(exp as usize, '.');
                    v.trim_end_matches('0').trim_end_matches('.').to_string()
                }
            } else {
                let zeros = exp.abs_diff(0) as usize;
                let mut x = "0.".to_string() + &"0".repeat(zeros) + s.trim_end_matches('0');
                if let Some(digits) = digits {
                    x.truncate(max(digits, zeros + 1));
                }
                x
            }
        } else if self.is_zero() {
            s
        } else {
            return s;
        };
        format!("{}{}{s}", if sign { "-" } else { "" }, radix.prefix())
    }

    fn digits(&self, radix: Radix) -> usize {
        let (_sign, s, exp) = self.to_sign_string_exp(radix.into(), None);
        s.len() + exp.unwrap_or_default().unsigned_abs() as usize
    }

    fn to_string_width(&self, radix: Radix, width: Option<usize>) -> String {
        let width = width.map(|w| {
            if w == 0 {
                1
            } else {
                let l = self.digits(radix);
                let w = w - 1 - radix.prefix().len() - if self.is_sign_negative() { 1 } else { 0 };
                min(w, l)
            }
        });
        self.to_string_scalar_len(radix, width)
    }
}

impl RpnToStringScalar for Rational {
    fn to_string_scalar_len(&self, radix: Radix, _digits: Option<usize>) -> String {
        let sign = match self.cmp0() {
            Ordering::Less => "-",
            Ordering::Equal => return "0".to_string(),
            Ordering::Greater => "",
        };
        if self.is_integer() {
            format!(
                "{sign}{}{}",
                radix.prefix(),
                self.clone().abs().to_string_radix(radix.into())
            )
        } else {
            format!(
                "{sign}{r}{}/{r}{}",
                self.numer().clone().abs().to_string_radix(radix.into()),
                self.denom().clone().abs().to_string_radix(radix.into()),
                r = radix.prefix()
            )
        }
    }
}

#[derive(Debug, Default, strum_macros::Display, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Radix {
    #[default]
    Decimal,
    Hex,
    Binary,
    Octal,
    // Base twelve
    Duodecimal,
}

impl Radix {
    pub(crate) fn prefix(&self) -> &str {
        match self {
            Radix::Decimal => "",
            Radix::Hex => "0x",
            Radix::Binary => "0b",
            Radix::Octal => "0o",
            Radix::Duodecimal => "0z",
        }
    }
    pub(crate) fn from_prefix(val: &str) -> Self {
        match val.to_lowercase().as_str() {
            "0b" | "b" => Self::Binary,
            "0o" | "o" => Self::Octal,
            "0d" | "d" => Self::Decimal,
            "0x" | "x" => Self::Hex,
            "0z" | "z" => Self::Duodecimal,
            _ => Self::default(),
        }
    }

    // (is_neg, Radix, rest_of_string)
    pub(crate) fn split_string(value: &str) -> (bool, Self, String) {
        let re = Regex::new(r"^(-)?(0[xXoObBdDzZ])?(.*)").unwrap();
        let value = value.trim_start_matches('+');
        if let Some(caps) = re.captures(value) {
            (
                caps.get(1).is_some(),
                caps.get(2)
                    .map(|x| Radix::from_prefix(x.as_str()))
                    .unwrap_or_default(),
                caps[3].to_string(),
            )
        } else {
            (false, Self::default(), value.to_string())
        }
    }
}

impl From<Radix> for i32 {
    fn from(r: Radix) -> Self {
        match r {
            Radix::Decimal => 10,
            Radix::Hex => 16,
            Radix::Binary => 2,
            Radix::Octal => 8,
            Radix::Duodecimal => 12,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, strum_macros::Display, PartialEq, Eq)]
pub(crate) enum Angle {
    #[default]
    Degree,
    Radian,
    Gradian,
}

impl From<i32> for Scalar {
    fn from(x: i32) -> Self {
        Scalar::Int(Rational::from(x))
    }
}

impl From<usize> for Scalar {
    fn from(x: usize) -> Self {
        Scalar::Int(Rational::from(x))
    }
}

impl From<Integer> for Scalar {
    fn from(x: Integer) -> Self {
        Scalar::Int(Rational::from(x))
    }
}

impl From<Rational> for Scalar {
    fn from(x: Rational) -> Self {
        Scalar::Int(x)
    }
}

impl From<Float> for Scalar {
    fn from(x: Float) -> Self {
        // Attempt to make Rational, but only if we're not out at
        // FLOAT_PRECISION which means it's a floating point rounding
        // error most likely
        if let Some(r) = x.to_rational()
            && r.denom() < &(Integer::from(1) << (FLOAT_PRECISION / 4))
        {
            Scalar::Int(r)
        } else {
            Scalar::Float(x)
        }
    }
}

impl From<Complex> for Scalar {
    fn from(x: Complex) -> Self {
        if x.imag().is_zero() {
            Scalar::from(x.real().clone())
        } else {
            Scalar::Complex(x)
        }
    }
}

fn parse_float(val: &str) -> Result<Float, String> {
    if val.is_empty() {
        return Ok(Float::with_val(FLOAT_PRECISION, 0));
    }
    let (f, r, v) = Radix::split_string(val);
    let v = Float::parse_radix(format!("{}{v}", if f { "-" } else { "" }), r.into())
        .map_err(|e| format!("{v}: {e}"))?;
    Ok(Float::with_val(FLOAT_PRECISION, v))
}

// Take finite floating point number and make into rational
fn parse_real(val: &str) -> Result<Rational, String> {
    if val.is_empty() {
        return Ok(Rational::zero());
    }
    if let Some((int, real)) = val.split_once('.') {
        let r_count = real.len() as u32;
        let all_int = int.to_string() + real;
        let (f, r, v) = Radix::split_string(&all_int);
        let v = Integer::parse_radix(format!("{}{v}", if f { "-" } else { "" }), r.into())
            .map_err(|e| format!("{v}: {e}"))?;

        Ok(Rational::from((
            v.complete(),
            Integer::from(i32::from(r)).pow(r_count),
        )))
    } else {
        parse_int(val).map(|x| Rational::from((x, 1)))
    }
}

fn parse_int(val: &str) -> Result<Integer, String> {
    let (f, r, v) = Radix::split_string(val);
    let v = Integer::parse_radix(format!("{}{v}", if f { "-" } else { "" }), r.into())
        .map_err(|e| format!("{v}: {e}"))?;
    Ok(v.complete())
}

fn dms2scalar(x1: Option<&Scalar>, x2: Option<&Scalar>, x3: Option<&Scalar>) -> Scalar {
    x1.cloned().unwrap_or_default() % 360.into()
        + x2.cloned().unwrap_or_default() % 60.into() / 60.into()
        + x3.cloned().unwrap_or_default() % 60.into() / 3600.into()
}

fn scalar2dms(xs: &Scalar) -> Vec<Scalar> {
    let mut m = vec![Scalar::zero(), Scalar::zero(), Scalar::zero()];
    m[0] = xs.clone().trunc() % Scalar::from(360);
    let f = xs.abs_sub(&m[0]) * Scalar::from(60);
    m[1] = f.clone().trunc();
    let f = (f - m[1].clone()) * Scalar::from(60);
    m[2] = f;

    m
}

fn img2float(v: &str) -> String {
    let v = v.trim().trim_end_matches('i').trim();
    match v {
        "" | "+" => "1".to_string(),
        "-" => "-1".to_string(),
        x => x.to_string(),
    }
}

fn median(mut list: Vec<Scalar>) -> Scalar {
    let len = list.len();

    let n = len >> 1;

    if len & 1 == 1 {
        let (_, val, _) = list.select_nth_unstable_by(n, |a, b| a.total_cmp(b));
        val.clone()
    } else {
        list.sort_unstable_by(|a, b| a.total_cmp(b));
        (list[n - 1].clone() + list[n].clone()) / 2.into()
    }
}

impl TryFrom<&str> for Scalar {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.contains('i') {
            let value = value.trim().trim_start_matches('+');
            let (real, imag) = if let Some((r, i)) = value.rsplit_once('+') {
                (r.trim(), img2float(i))
            } else if let Some((r, i)) = value.rsplit_once('-') {
                let i = format!("-{}", i.trim());
                (r.trim(), img2float(&i))
            } else {
                ("0", img2float(value))
            };

            let r = parse_float(real)?;
            let i = parse_float(&imag)?;

            Ok(Scalar::from(Complex::from((r, i))))
        } else if value.contains('[') {
            Err("Parsing Matrix as Scalar".to_string())
        } else if value.contains('.') {
            Ok(Scalar::from(parse_real(value)?))
        } else if let Some((numer, denom)) = value.split_once('/') {
            let n = parse_int(numer)?;
            let d = parse_int(denom)?;

            Ok(Scalar::from(Rational::from((n, d))))
        } else {
            let v = parse_int(value)?;
            Ok(Scalar::from(Rational::from(v)))
        }
    }
}

impl TryFrom<&str> for Value {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.contains('[') {
            // [ 1 2 3; 4 5 6; 1+2i 2.4 1.1-3.12i ]
            let v: Vec<Vec<&str>> = value
                .split(&[';', '[', ']'][..])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.split(' ').collect())
                .collect();

            let rows = v.len();
            let cols = v[0].len();
            Matrix::from_vec(
                rows,
                cols,
                v.into_iter()
                    .flatten()
                    .map(Scalar::try_from)
                    .collect::<Result<Vec<Scalar>, String>>()?,
            )
            .map(|m| m.into())
            .map_err(|e| e.to_string())
        } else if value.contains('(') {
            Ok(value
                .split(&['(', ')', ' '][..])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(Scalar::try_from)
                .collect::<Result<Vec<Scalar>, String>>()?
                .into())
        } else {
            Scalar::try_from(value).map(|s| s.into())
        }
    }
}

impl Num for Scalar {
    type FromStrRadixErr = String;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Rational::parse_radix(str, radix as i32)
            .map(|v| Scalar::from(Rational::from(v)))
            .map_err(|e| e.to_string())
    }
}

impl Scalar {
    /// Exponent (`e^x`)
    pub(crate) fn exp(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).exp()),
            Scalar::Float(x) => Scalar::from(x.exp()),
            Scalar::Complex(x) => Scalar::from(x.exp()),
        }
    }

    /// Natural Logarithm
    pub(crate) fn ln(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).ln()),
            Scalar::Float(x) => Scalar::from(x.ln()),
            Scalar::Complex(x) => Scalar::from(x.ln()),
        }
    }

    /// Logarithm (base 10)
    pub(crate) fn log10(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).log10()),
            Scalar::Float(x) => Scalar::from(x.log10()),
            Scalar::Complex(x) => Scalar::from(x.log10()),
        }
    }

    /// Logarithm (base 2)
    pub(crate) fn log2(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).log2()),
            Scalar::Float(x) => Scalar::from(x.log2()),
            Scalar::Complex(x) => {
                Scalar::from(x.log10() / Complex::with_val(FLOAT_PRECISION, (2, 0)).log10())
            }
        }
    }

    /// Logarithm (base N)
    pub(crate) fn log_n(self, n: Self) -> Self {
        match (self, n) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(
                Float::with_val(FLOAT_PRECISION, a).log10()
                    / Float::with_val(FLOAT_PRECISION, b).log10(),
            ),
            (Scalar::Int(a), Scalar::Float(b)) => {
                Scalar::from(Float::with_val(FLOAT_PRECISION, a).log10() / b.log10())
            }
            (Scalar::Int(a), Scalar::Complex(b)) => {
                Scalar::from(Float::with_val(FLOAT_PRECISION, a).log10() / b.log10())
            }

            (Scalar::Float(a), Scalar::Int(b)) => {
                Scalar::from(a.log10() / Float::with_val(FLOAT_PRECISION, b).log10())
            }
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a.log10() / b.log10()),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a.log10() / b.log10()),

            (Scalar::Complex(a), Scalar::Int(b)) => {
                Scalar::from(a.log10() / Float::with_val(FLOAT_PRECISION, b).log10())
            }
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a.log10() / b.log10()),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a.log10() / b.log10()),
        }
    }

    /// Factorial (`n!`)
    pub(crate) fn try_factorial(self) -> Result<Self, String> {
        match self {
            Scalar::Int(x) if x.is_integer() => x.numer().factorial().map(|x| x.into()),
            _ => Err("n! only implemented for integers".to_string()),
        }
    }

    /// Permutation (`nPm`)
    /// `P(n, m) = n! / (n - m)!`
    pub(crate) fn try_permute(self, r: Self) -> Result<Self, String> {
        if self < r {
            return Err("P(n, m) only valid for n < m".to_string());
        }
        match (self, r) {
            (Scalar::Int(n), Scalar::Int(r))
                if n.is_integer() && !n.is_negative() && r.is_integer() && !r.is_negative() =>
            {
                let n = n.numer();
                let r = r.numer();

                if n == r {
                    n.factorial().map(|x| x.into())
                } else {
                    let mut val = Integer::one();
                    let mut i = Integer::one() + n - r;
                    while i <= *n {
                        val *= i.clone();
                        i += 1;
                    }
                    Ok(val.into())
                }
            }
            _ => Err("n! only implemented for positive integers".to_string()),
        }
    }

    /// Combination (`nCm`)
    /// `C(n, m) = n! / m!(n - m)!`
    pub(crate) fn try_choose(self, r: Self) -> Result<Self, String> {
        if self < r {
            return Err("C(n, m) only valid for n < m".to_string());
        }
        match (self, r) {
            (Scalar::Int(n), Scalar::Int(r))
                if n.is_integer() && !n.is_negative() && r.is_integer() && !r.is_negative() =>
            {
                let n = n.numer();
                let r = r.numer();

                if n == r || r.is_zero() {
                    Ok(1.into())
                } else {
                    let mut val = Integer::one();
                    let mut i = Integer::one() + n - r;
                    while i <= *n {
                        val *= i.clone();
                        i += 1;
                    }
                    Ok(Scalar::from(val / r.factorial()?))
                }
            }
            _ => Err("n! only implemented for natural numbers".to_string()),
        }
    }

    /// Factor Integer
    pub(crate) fn factor(self) -> Result<Vec<Self>, String> {
        match self {
            Scalar::Int(x) => {
                if x < 0 {
                    Err("Cannot Factor Negative Number".to_string())
                } else if !x.is_integer() {
                    Err("Cannot Factor Rational Number".to_string())
                } else if x == 1 {
                    Ok(vec![x.into()])
                } else {
                    let mut val = x.numer().clone();
                    let mut factor = Integer::from(2);
                    let mut factors = Vec::new();
                    while val >= factor {
                        if val.is_divisible(&factor) {
                            factors.push(factor.clone().into());
                            val /= &factor;
                        } else {
                            factor.next_prime_mut();
                        }
                    }
                    Ok(factors)
                }
            }
            Scalar::Float(_) => Err("Cannot Factor Floating point".to_string()),
            Scalar::Complex(_) => Err("No multiplication of tuples".to_string()), // @@
        }
    }

    /// Floor
    pub(crate) fn floor(self) -> Self {
        match self {
            Scalar::Int(x) => x.floor().into(),
            Scalar::Float(x) => x.floor().into(),
            Scalar::Complex(x) => {
                Complex::from((x.real().clone().floor(), x.imag().to_owned().floor())).into()
            }
        }
    }

    /// Floor
    pub(crate) fn ceil(self) -> Self {
        match self {
            Scalar::Int(x) => x.ceil().into(),
            Scalar::Float(x) => x.ceil().into(),
            Scalar::Complex(x) => {
                Complex::from((x.real().clone().ceil(), x.imag().to_owned().ceil())).into()
            }
        }
    }

    /// Truncate to Integer
    pub(crate) fn trunc(self) -> Self {
        match self {
            Scalar::Int(x) => x.trunc().into(),
            Scalar::Float(x) => x.trunc().into(),
            Scalar::Complex(x) => {
                Complex::from((x.real().clone().trunc(), x.imag().to_owned().trunc())).into()
            }
        }
    }

    /// Round to nearest Integer
    pub(crate) fn round(self) -> Self {
        match self {
            Scalar::Int(x) => x.round().into(),
            Scalar::Float(x) => x.round().into(),
            Scalar::Complex(x) => {
                Complex::from((x.real().clone().round(), x.imag().to_owned().round())).into()
            }
        }
    }

    /// Return the numerator (as `usize`) iff denominator is 1
    fn get_usize(self) -> Option<usize> {
        match self {
            Scalar::Int(x) => x.is_integer().then(|| x.numer().to_usize()).flatten(),
            _ => None,
        }
    }

    /// Return the numerator iff denominator is 1
    fn get_integer(&self) -> Option<Integer> {
        match self {
            Scalar::Int(x) => x.is_integer().then(|| x.numer().clone()),
            _ => None,
        }
    }

    /// Implement `total_cmp`
    fn total_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => a.cmp(b),
            (Scalar::Int(a), Scalar::Float(b)) => {
                let fa = Float::with_val(FLOAT_PRECISION, a);
                fa.total_cmp(b)
            }
            (Scalar::Int(a), Scalar::Complex(b)) => {
                let ca = Complex::with_val(FLOAT_PRECISION, a);
                ca.total_cmp(b)
            }

            (Scalar::Float(a), Scalar::Int(b)) => a.total_cmp(&Float::with_val(FLOAT_PRECISION, b)),
            (Scalar::Float(a), Scalar::Float(b)) => a.total_cmp(b),
            (Scalar::Float(a), Scalar::Complex(b)) => {
                let ca = Complex::with_val(FLOAT_PRECISION, a);
                ca.total_cmp(b)
            }

            (Scalar::Complex(a), Scalar::Int(b)) => {
                a.total_cmp(&Complex::with_val(FLOAT_PRECISION, b))
            }
            (Scalar::Complex(a), Scalar::Float(b)) => {
                a.total_cmp(&Complex::with_val(FLOAT_PRECISION, b))
            }
            (Scalar::Complex(a), Scalar::Complex(b)) => a.total_cmp(b),
        }
    }

    // Trig Functions
    pub(crate) fn sin(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.sin().into(),
                    Angle::Degree => f.sin_u(360).into(),
                    Angle::Gradian => f.sin_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.sin().into(),
                Angle::Degree => f.sin_u(360).into(),
                Angle::Gradian => f.sin_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.sin().into(),
                Angle::Degree => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 180))
                .sin()
                .into(),
                Angle::Gradian => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 200))
                .sin()
                .into(),
            },
        }
    }

    pub(crate) fn cos(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.cos().into(),
                    Angle::Degree => f.cos_u(360).into(),
                    Angle::Gradian => f.cos_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.cos().into(),
                Angle::Degree => f.cos_u(360).into(),
                Angle::Gradian => f.cos_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.cos().into(),
                Angle::Degree => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 180))
                .cos()
                .into(),
                Angle::Gradian => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 200))
                .cos()
                .into(),
            },
        }
    }

    pub(crate) fn tan(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.tan().into(),
                    Angle::Degree => f.tan_u(360).into(),
                    Angle::Gradian => f.tan_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.tan().into(),
                Angle::Degree => f.tan_u(360).into(),
                Angle::Gradian => f.tan_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.tan().into(),
                Angle::Degree => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 180))
                .tan()
                .into(),
                Angle::Gradian => (x * Complex::with_val(FLOAT_PRECISION, Constant::Pi)
                    / Complex::with_val(FLOAT_PRECISION, 200))
                .tan()
                .into(),
            },
        }
    }

    pub(crate) fn asin(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.asin().into(),
                    Angle::Degree => f.asin_u(360).into(),
                    Angle::Gradian => f.asin_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.asin().into(),
                Angle::Degree => f.asin_u(360).into(),
                Angle::Gradian => f.asin_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.asin().into(),
                Angle::Degree => (x.asin() * Complex::with_val(FLOAT_PRECISION, 180)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
                Angle::Gradian => (x.asin() * Complex::with_val(FLOAT_PRECISION, 200)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
            },
        }
    }

    pub(crate) fn acos(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.acos().into(),
                    Angle::Degree => f.acos_u(360).into(),
                    Angle::Gradian => f.acos_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.acos().into(),
                Angle::Degree => f.acos_u(360).into(),
                Angle::Gradian => f.acos_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.acos().into(),
                Angle::Degree => (x.acos() * Complex::with_val(FLOAT_PRECISION, 180)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
                Angle::Gradian => (x.acos() * Complex::with_val(FLOAT_PRECISION, 200)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
            },
        }
    }

    pub(crate) fn atan(self, angle: Angle) -> Self {
        match self {
            Scalar::Int(x) => {
                let f = Float::with_val(FLOAT_PRECISION, x);
                match angle {
                    Angle::Radian => f.atan().into(),
                    Angle::Degree => f.atan_u(360).into(),
                    Angle::Gradian => f.atan_u(400).into(),
                }
            }
            Scalar::Float(f) => match angle {
                Angle::Radian => f.atan().into(),
                Angle::Degree => f.atan_u(360).into(),
                Angle::Gradian => f.atan_u(400).into(),
            },
            Scalar::Complex(x) => match angle {
                Angle::Radian => x.atan().into(),
                Angle::Degree => (x.atan() * Complex::with_val(FLOAT_PRECISION, 180)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
                Angle::Gradian => (x.atan() * Complex::with_val(FLOAT_PRECISION, 200)
                    / Complex::with_val(FLOAT_PRECISION, Constant::Pi))
                .into(),
            },
        }
    }

    // @@ Floats don't get exactly the right length
    pub(crate) fn to_string_radix(
        &self,
        radix: Radix,
        rational: bool,
        width: Option<usize>,
    ) -> String {
        //eprintln!("{self:?} digits:{width:?}");
        match self {
            Scalar::Int(x) => {
                if !rational && !x.is_integer() {
                    Float::with_val(FLOAT_PRECISION, x).to_string_width(radix, width)
                } else {
                    x.to_string_scalar(radix)
                }
            }
            Scalar::Float(x) => x.to_string_width(radix, width),
            Scalar::Complex(x) => {
                let r = x.real();
                let i = x.imag();
                let sign = if i.is_sign_positive() { "+" } else { "" };
                if r.is_zero() {
                    format!("{}i", i.to_string_width(radix, width.map(|x| x - 1)))
                } else if let Some(width) = width {
                    let rlen = r.to_string_scalar(radix).len();
                    let ilen = i.to_string_scalar(radix).len();

                    let digits = width - 2 - radix.prefix().len() * 2;
                    let (rlen, ilen) = if digits > rlen + ilen {
                        (rlen, ilen)
                    } else if rlen >= digits && ilen >= digits {
                        let x = digits / 2;
                        (digits - x, x)
                    } else {
                        let t = rlen + ilen;
                        //eprintln!("digits:{digits} t:{t} :: {r} ({rlen}) + {i} ({ilen}) i");
                        (max(2, rlen * digits / t) - 1, max(2, ilen * digits / t) - 1)
                    };
                    //eprintln!("width:{width} digits:{digits} rlen:{rlen} ilen:{ilen}");
                    format!(
                        "{}{}{}i",
                        r.to_string_scalar_len(radix, Some(rlen)),
                        sign,
                        i.to_string_scalar_len(radix, Some(ilen))
                    )
                } else {
                    format!(
                        "{}{}{}i",
                        r.to_string_scalar(radix),
                        sign,
                        i.to_string_scalar(radix)
                    )
                }
            }
        }
    }
}

impl One for Scalar {
    fn one() -> Self {
        Scalar::Int(Rational::one())
    }
}

impl Zero for Scalar {
    fn zero() -> Self {
        Scalar::Int(Rational::zero())
    }

    /// True if value is Zero
    fn is_zero(&self) -> bool {
        match self {
            Scalar::Int(x) => x.is_zero(),
            Scalar::Float(x) => x.is_zero(),
            Scalar::Complex(x) => x.is_zero(),
        }
    }
}

impl Value {
    /// Return the numerator iff denominator is 1
    fn get_integer(&self) -> Option<Integer> {
        match self {
            Value::Scalar(Scalar::Int(x)) => x.is_integer().then(|| x.numer().clone()),
            _ => None,
        }
    }

    /// Return the numerator (as `usize`) iff denominator is 1
    fn get_usize(&self) -> Option<usize> {
        match self {
            Value::Scalar(Scalar::Int(x)) => x.is_integer().then(|| x.numer().to_usize()).flatten(),
            _ => None,
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        match self {
            Value::Scalar(x) => x.is_zero(),
            Value::Tuple(x) => x.is_empty(),
            Value::Matrix(x) => {
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        if !x[i][j].is_zero() {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    pub(crate) fn try_sqrt(self) -> Result<Self, String> {
        self.try_root(2.into())
    }

    pub(crate) fn to_string_radix(
        &self,
        radix: Radix,
        rational: bool,
        flat: bool,
        digits: impl Into<Option<usize>>,
    ) -> String {
        let digits = digits.into();
        match self {
            Value::Scalar(x) => x.to_string_radix(radix, rational, digits),
            Value::Tuple(tuple) => {
                if let Some(digits) = digits {
                    let mut digits = digits - 3;
                    let mut len = tuple.len();

                    let mut output = "( ".to_owned();
                    for x in tuple {
                        let n = (digits / len) - 1;
                        let s = x.to_string_radix(radix, rational, Some(n));

                        output += &s;
                        output += " ";

                        digits -= s.len();
                        len -= 1;
                    }
                    output.push(')');

                    output
                } else {
                    format!(
                        "( {} )",
                        tuple
                            .iter()
                            .map(|x| x.to_string_radix(radix, rational, None))
                            .join(" ")
                    )
                }
            }
            Value::Matrix(x) => {
                let mut output = "[ ".to_owned();
                let rowmax = x.rows();
                // items per row
                let mut len = if flat { rowmax * x.cols() } else { x.cols() };

                // Total row digits - 1 space per item - len("[ ") - 2 per row (for ";" or "]")
                let mut row_digits = digits.map(|x| x - len - 2 - if flat { rowmax } else { 1 });

                let minwidth = row_digits.map(|x| max(x / len, 2)).unwrap_or_default();

                for i in 0..rowmax {
                    for j in 0..x.cols() {
                        let n = row_digits.map(|x| max(x / len, minwidth));
                        let s = x[i][j].to_string_radix(radix, rational, n);

                        output += &s;
                        output += " ";

                        len -= 1;
                        if let Some(x) = row_digits {
                            if x > s.len() {
                                row_digits = Some(x - s.len());
                            } else {
                                row_digits = Some(0);
                            }
                        }
                    }

                    output += if i == rowmax - 1 {
                        "]"
                    } else if flat {
                        "; "
                    } else {
                        len = x.cols();
                        row_digits = digits.map(|x| x - 2 - len);
                        ";\n"
                    };
                }
                output
            }
        }
    }

    pub(crate) fn lines(&self) -> usize {
        match self {
            Value::Scalar(_) => 1,
            Value::Tuple(_) => 1, // FIXME: wrap?
            Value::Matrix(x) => x.rows(),
        }
    }

    pub(crate) fn try_factor(self) -> Result<Value, String> {
        match self {
            Value::Scalar(x) => x.factor().map(|x| x.into()),
            Value::Tuple(_) => Err("Factoring Tuple not supported".to_string()),
            Value::Matrix(_) => Err("No prime factors of matricies".to_string()),
        }
    }

    pub(crate) fn try_factorial(self) -> Result<Value, String> {
        match self {
            Value::Scalar(x) => Ok(x.try_factorial()?.into()),
            _ => Err(format!("{self:?}! is not INT!")),
        }
    }

    pub(crate) fn try_permute(self, r: Self) -> Result<Value, String> {
        match (self, r) {
            (Value::Scalar(x), Value::Scalar(r)) => Ok(x.try_permute(r)?.into()),
            _ => Err("Permuation only implement for INT P INT".to_string()),
        }
    }

    pub(crate) fn try_choose(self, r: Self) -> Result<Value, String> {
        match (self, r) {
            (Value::Scalar(x), Value::Scalar(r)) => Ok(x.try_choose(r)?.into()),
            _ => Err("Combination only implement for INT C INT".to_string()),
        }
    }

    pub(crate) fn try_modulo(&self, b: &Value) -> Result<Value, String> {
        match (self, b) {
            (_, b) if b.is_zero() => Err("Modulo by zero".to_string()),
            (_, Value::Matrix(_)) => Err("Modulo by Matrix".to_string()),
            (_, Value::Tuple(_)) => Err("Modulo by Tuple".to_string()),
            (_, Value::Scalar(b)) if b.is_negative() => Err("Modulo by negative".to_string()),
            (Value::Tuple(_), Value::Scalar(_)) => Err("NYI: Modulo of Tuple".to_string()),
            (Value::Matrix(_), Value::Scalar(_)) => Err("NYI: Modulo of Matrix".to_string()),
            (Value::Scalar(a), Value::Scalar(b)) => {
                if a.is_positive() {
                    Ok(Value::from(a.clone() % b.clone()))
                } else if let (Some(ai), Some(bi)) = (self.get_integer(), b.get_integer()) {
                    Ok(Value::from(ai.rem_euc(bi)))
                } else {
                    Err("Modulo of non-integer".to_string())
                }
            }
        }
    }

    pub(crate) fn try_root(self, other: Value) -> Result<Self, String> {
        self.pow(other.inv()?)
    }

    pub(crate) fn try_ln(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.ln())),
            _ => Err("NYI".to_string()),
        }
    }

    pub(crate) fn try_abs(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.abs())),
            _ => Err("NYI".to_string()),
        }
    }

    pub(crate) fn try_exp(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.exp())),
            Value::Tuple(_) => Err("NYI".to_string()),
            Value::Matrix(x) => x.try_exp().map(Value::Matrix),
        }
    }

    pub(crate) fn try_log10(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.log10())),
            Value::Tuple(_) => Err("NYI".to_string()),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub(crate) fn try_log2(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.log2())),
            Value::Tuple(_) => Err("NYI".to_string()),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub(crate) fn try_log_n(self, other: Self) -> Result<Self, String> {
        if let (Value::Scalar(x), Value::Scalar(y)) = (self, other) {
            Ok(Value::Scalar(x.log_n(y)))
        } else {
            Err("NYI".to_string())
        }
    }

    pub(crate) fn try_dms_conv(&self) -> Result<Self, String> {
        match &self {
            Value::Scalar(x) => Ok(scalar2dms(x).into()),
            Value::Tuple(t) => {
                if t.len() > 3 {
                    Err(format!("Tuple too long ({} > 3)", t.len()))
                } else {
                    Ok(dms2scalar(t.front(), t.get(1), t.get(2)).into())
                }
            }
            Value::Matrix(m) => {
                if m.cols() > 3 {
                    Err(format!(
                        "Matrix of incorrect size [{}x{}] expected [1x3] or [2x3]",
                        m.rows(),
                        m.cols()
                    ))
                } else {
                    match m.rows() {
                        1 => Ok(dms2scalar(m[0].first(), m[0].get(1), m[0].get(2)).into()),
                        2 if m.cols() > 1 => Ok(libmat::matrix! {
                            { dms2scalar(m[0].first(), m[0].get(1), m[0].get(2)) },
                            { dms2scalar(m[1].first(), m[1].get(1), m[1].get(2)) }
                        }
                        .into()),
                        2 => Matrix::from_vec(
                            2,
                            3,
                            scalar2dms(&m[0][0])
                                .into_iter()
                                .chain(scalar2dms(&m[1][0]))
                                .collect::<Vec<_>>(),
                        )
                        .map(|x| x.into())
                        .map_err(|e| format!("Failed to build 2x3 matrix: {e}")),
                        _ => Err(format!(
                            "Matrix of incorrect size [{}x{}] expected [1x3] or [2x3]",
                            m.rows(),
                            m.cols()
                        )),
                    }
                }
            }
        }
    }

    /// Truncate values to Integer
    pub(crate) fn trunc(self) -> Self {
        match self {
            Value::Scalar(x) => x.trunc().into(),
            Value::Tuple(x) => x.into_iter().map(|x| x.trunc()).collect::<Vec<_>>().into(),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].clone().trunc();
                    }
                }
                m.into()
            }
        }
    }

    /// Truncate values to Integer
    pub(crate) fn floor(self) -> Self {
        match self {
            Value::Scalar(x) => x.floor().into(),
            Value::Tuple(x) => x.into_iter().map(|x| x.floor()).collect::<Vec<_>>().into(),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].clone().floor();
                    }
                }
                m.into()
            }
        }
    }

    /// Truncate values to Integer
    pub(crate) fn ceil(self) -> Self {
        match self {
            Value::Scalar(x) => x.ceil().into(),
            Value::Tuple(x) => x.into_iter().map(|x| x.ceil()).collect::<Vec<_>>().into(),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].clone().ceil();
                    }
                }
                m.into()
            }
        }
    }

    /// Round values to Integer
    pub(crate) fn round(self) -> Self {
        match self {
            Value::Scalar(x) => x.round().into(),
            Value::Tuple(x) => x.into_iter().map(|x| x.round()).collect::<Vec<_>>().into(),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].clone().round();
                    }
                }
                m.into()
            }
        }
    }

    // Trig Functions
    pub(crate) fn try_sin(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.sin(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    pub(crate) fn try_cos(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.cos(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    pub(crate) fn try_tan(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.tan(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    pub(crate) fn try_asin(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.asin(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    pub(crate) fn try_acos(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.acos(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    pub(crate) fn try_atan(self, angle: Angle) -> Result<Self, String> {
        match self {
            Value::Scalar(s) => Ok(Value::Scalar(s.atan(angle))),
            Value::Tuple(_t) => Err("NYI".to_owned()),
            Value::Matrix(_m) => Err("NYI".to_owned()),
        }
    }

    // Tuple Functions
    pub(crate) fn try_push(self, b: Self) -> Result<Self, String> {
        match (self, b) {
            (Value::Scalar(s), Value::Tuple(mut t)) => {
                t.push_front(s);
                Ok(t.into())
            }
            (Value::Tuple(mut t), Value::Scalar(s)) => {
                t.push_back(s);
                Ok(t.into())
            }
            (Value::Tuple(mut t1), Value::Tuple(mut t2)) => {
                t1.append(&mut t2);
                Ok(t1.into())
            }
            (Value::Scalar(a), Value::Scalar(b)) => Ok(vec![a, b].into()),
            (Value::Matrix(_), _) | (_, Value::Matrix(_)) => {
                Err("Unsupported: modification of matrices".to_string())
            }
        }
    }

    pub(crate) fn try_unpush(self) -> Result<Vec<Self>, String> {
        match self {
            Value::Scalar(_) => Err("Illegal Operation: unpush of scalar".to_string()),
            Value::Tuple(mut t) => {
                if let Some(a) = t.pop_front() {
                    Ok(vec![t.into(), a.into()])
                } else {
                    Ok(vec![t.into()])
                }
            }
            Value::Matrix(_) => Err("Unsupported: modification of matrices".to_string()),
        }
    }

    pub(crate) fn try_pull(self) -> Result<Vec<Self>, String> {
        match self {
            Value::Scalar(s) => Ok(vec![Value::Scalar(s)]),
            Value::Tuple(mut t) => {
                if let Some(a) = t.pop_back() {
                    Ok(vec![a.into(), t.into()])
                } else {
                    Ok(vec![t.into()])
                }
            }
            Value::Matrix(_) => Err("Unsupported: modification of matrices".to_string()),
        }
    }

    pub(crate) fn try_expand(self) -> Result<Vec<Self>, String> {
        match self {
            Value::Scalar(s) => Ok(vec![Value::Scalar(s)]),
            Value::Tuple(t) => Ok(t.into_iter().rev().map(Value::Scalar).collect()),
            Value::Matrix(m) if m.rows() == 1 => {
                Ok(m[0].iter().rev().cloned().map(Value::Scalar).collect())
            }
            _ => Err("Unable to expand".to_string()),
        }
    }

    pub(crate) fn sum(self) -> Self {
        match self {
            Value::Scalar(s) => Value::Scalar(s),
            Value::Tuple(t) => Value::Scalar(t.into_iter().sum()),
            Value::Matrix(m) => Value::Scalar(m.into_iter().sum()),
        }
    }

    pub(crate) fn product(self) -> Self {
        match self {
            Value::Scalar(s) => Value::Scalar(s),
            Value::Tuple(t) => Value::Scalar(t.into_iter().product()),
            Value::Matrix(m) => Value::Scalar(m.into_iter().product()),
        }
    }

    // Stats Functions
    pub(crate) fn mean(self) -> Scalar {
        match self {
            Value::Scalar(s) => s,
            Value::Tuple(t) => {
                let len = t.len().into();
                let v: Scalar = t.into_iter().sum();
                v / len
            }
            Value::Matrix(m) => {
                let len: Scalar = Scalar::from(m.rows()) * m.cols().into();
                let sum: Scalar = m.into_iter().sum();
                sum / len
            }
        }
    }

    /// Geometric Mean
    /// n-th root of the product of all the values
    pub(crate) fn geometric_mean(self) -> Scalar {
        match self {
            Value::Scalar(s) => s,
            Value::Tuple(t) => {
                let len = t.len().into();
                let v: Scalar = t.into_iter().map(|x| x.ln()).sum();
                (v / len).exp()
            }
            Value::Matrix(m) => {
                let len: Scalar = Scalar::from(m.rows()) * m.cols().into();
                let sum: Scalar = m.into_iter().map(|x| x.ln()).sum();
                (sum / len).exp()
            }
        }
    }

    /// Harmonic Mean
    pub(crate) fn harmonic_mean(self) -> Scalar {
        match self {
            Value::Scalar(s) => s,
            Value::Tuple(t) => {
                let len: Scalar = t.len().into();
                let v: Scalar = t.into_iter().map(|x| x.inv()).sum();
                len / v
            }
            Value::Matrix(m) => {
                let len: Scalar = Scalar::from(m.rows()) * m.cols().into();
                let sum: Scalar = m.into_iter().map(|x| x.inv()).sum();
                len / sum
            }
        }
    }

    pub(crate) fn median(self) -> Scalar {
        match self {
            Value::Scalar(s) => s,
            Value::Tuple(t) => median(t.into_iter().collect()),
            Value::Matrix(m) => {
                let list = m.into_iter().collect::<Vec<_>>();
                median(list)
            }
        }
    }

    /// Geothmetic Meandian (XKCD #2435)
    ///
    /// F(x0, x1, ... xN) -> (Arithmatic Mean, Geometric Mean, Median)
    /// gmdn(...) -> F(F(F(F(...))))) until it converges
    ///
    /// This funcition computes a single iteration
    pub(crate) fn gmdn(self) -> Self {
        match self {
            Value::Scalar(s) => Value::Scalar(s),
            Value::Tuple(_) => Value::Tuple(
                [
                    self.clone().mean(),
                    self.clone().geometric_mean(),
                    self.median(),
                ]
                .into(),
            ),
            Value::Matrix(_) => Value::Tuple(
                [
                    self.clone().mean(),
                    self.clone().geometric_mean(),
                    self.median(),
                ]
                .into(),
            ),
        }
    }

    pub(crate) fn sort(self) -> Self {
        match self {
            Value::Scalar(s) => Value::Scalar(s),
            Value::Tuple(t) => Value::Tuple(
                t.into_iter()
                    .sorted_unstable_by(|a, b| a.total_cmp(b))
                    .collect(),
            ),
            Value::Matrix(m) => Value::Tuple(
                m.into_iter()
                    .sorted_unstable_by(|a, b| a.total_cmp(b))
                    .collect(),
            ),
        }
    }

    pub(crate) fn standard_deviation(self) -> Self {
        match self {
            Value::Scalar(_) => Value::Scalar(Scalar::zero()),
            Value::Tuple(t) => {
                let len: Scalar = t.len().into();
                let sum: Scalar = t.iter().cloned().sum();
                let mean = sum / len.clone();
                let differences: Scalar = t
                    .into_iter()
                    .map(|x| (x - mean.clone()).pow(2.into()))
                    .sum::<Scalar>();
                Value::Scalar((differences / len).pow(Scalar::from(2).inv()))
            }
            Value::Matrix(m) => {
                let len: Scalar = Scalar::from(m.rows()) * m.cols().into();
                let sum: Scalar = m.clone().into_iter().sum();
                let mean = sum / len.clone();
                let differences: Scalar = m
                    .into_iter()
                    .map(|x| (x - mean.clone()).pow(2.into()))
                    .sum::<Scalar>();
                Value::Scalar((differences / len).pow(Scalar::from(2).inv()))
            }
        }
    }

    // Matrix Only Functions
    pub(crate) fn try_det(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No determinate for Scalar".to_string()),
            Value::Tuple(_) => Err("No determinate for Tuple".to_string()),
            Value::Matrix(x) => Ok(Value::Scalar(x.det().map_err(|e| e.to_string())?)),
        }
    }

    pub(crate) fn try_rref(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No determinate for Scalar".to_string()),
            Value::Tuple(_) => Err("No determinate for Tuple".to_string()),
            Value::Matrix(x) => Ok(Value::Matrix(x.rref())),
        }
    }

    pub(crate) fn try_transpose(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No Transpose for Scalar".to_string()),
            Value::Tuple(t) => Ok(t.into_iter().rev().collect::<VecDeque<_>>().into()),
            Value::Matrix(x) => Ok(Value::Matrix(x.transpose())),
        }
    }

    // Constants
    pub(crate) fn identity(n: Value) -> Result<Self, String> {
        match n {
            Value::Scalar(n) => {
                if let Some(x) = n.get_usize() {
                    Matrix::one(x).map(Value::Matrix).map_err(|e| e.to_string())
                } else {
                    Err("Identity Matrix can only be created with integer size".to_string())
                }
            }
            Value::Tuple(_) => Err("Not INT or Matrix".to_string()),
            Value::Matrix(x) => {
                if x.is_square() {
                    Matrix::one(x.rows())
                        .map(Value::Matrix)
                        .map_err(|e| e.to_string())
                } else {
                    Err("Matrix not square, cannot convert to identity".to_string())
                }
            }
        }
    }

    pub(crate) fn ones(n: Value) -> Result<Self, String> {
        match n {
            Value::Scalar(n) => {
                if let Some(x) = n.get_usize() {
                    Matrix::new(x, x, Scalar::one())
                        .map(Value::Matrix)
                        .map_err(|e| e.to_string())
                } else {
                    Err("Identity Matrix can only be created with integer size".to_string())
                }
            }
            Value::Tuple(_) => Err("Not INT or Matrix".to_string()),
            Value::Matrix(x) => Matrix::new(x.rows(), x.cols(), Scalar::one())
                .map(Value::Matrix)
                .map_err(|e| e.to_string()),
        }
    }

    pub(crate) fn e() -> Self {
        let f = Float::with_val(FLOAT_PRECISION, 1);
        Value::Scalar(Scalar::from(f.exp()))
    }
    pub(crate) fn pi() -> Self {
        Value::Scalar(Scalar::from(Float::with_val(FLOAT_PRECISION, Constant::Pi)))
    }
    pub(crate) fn catalan() -> Self {
        Value::Scalar(Scalar::from(Float::with_val(
            FLOAT_PRECISION,
            Constant::Catalan,
        )))
    }
    pub(crate) fn avagadro() -> Self {
        let r: Scalar = Rational::from((602_214_076, 1)).into();
        let ten: Scalar = Rational::from((10, 1)).into();
        let ex: Scalar = Rational::from((23, 1)).into();
        Value::Scalar(r * ten.pow(ex))
    }
}

impl ops::Add<Value> for Value {
    type Output = Result<Self, String>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a + b).map(Value::Matrix).map_err(|e| e.to_string())
            }
            (Value::Tuple(a), Value::Tuple(b)) => Ok(a
                .into_iter()
                .zip_longest(b)
                .map(|x| match x {
                    EitherOrBoth::Both(x, y) => x + y,
                    EitherOrBoth::Left(x) => x,
                    EitherOrBoth::Right(x) => x,
                })
                .collect::<Vec<_>>()
                .into()),
            (Value::Tuple(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Tuple(_)) => {
                Err("NYI".to_string())
            }
            (Value::Scalar(_), _) | (_, Value::Scalar(_)) => {
                Err("Illegal Operation: Scalar and Non-Scalar Addition".to_string())
            }
        }
    }
}

impl ops::BitAnd<Value> for Value {
    type Output = Result<Value, String>;

    fn bitand(self, other: Self) -> Self::Output {
        if let (Value::Scalar(a), Value::Scalar(b)) = (self, other) {
            (a & b).map(Value::Scalar)
        } else {
            Err("NYI".to_string())
        }
    }
}

impl ops::BitOr<Value> for Value {
    type Output = Result<Value, String>;

    fn bitor(self, other: Self) -> Self::Output {
        if let (Value::Scalar(a), Value::Scalar(b)) = (self, other) {
            (a | b).map(Value::Scalar)
        } else {
            Err("NYI".to_string())
        }
    }
}

impl ops::BitXor<Value> for Value {
    type Output = Result<Value, String>;

    fn bitxor(self, other: Self) -> Self::Output {
        if let (Value::Scalar(a), Value::Scalar(b)) = (self, other) {
            (a ^ b).map(Value::Scalar)
        } else {
            Err("NYI".to_string())
        }
    }
}

impl ops::Div<Value> for Value {
    type Output = Result<Value, String>;

    // This does use x * 1/y which trips clippy
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self::Output {
        self * other.inv()?
    }
}

impl Inv for Value {
    type Output = Result<Value, String>;

    fn inv(self) -> Self::Output {
        match self {
            Value::Scalar(x) => {
                if x.is_zero() {
                    Err("Error: Division by zero".to_string())
                } else {
                    Ok(Value::Scalar(x.inv()))
                }
            }
            Value::Tuple(_) => Err("No multiplication of tuples".to_string()),
            Value::Matrix(x) => {
                if let Ok(Some(m)) = x.inv() {
                    Ok(Value::Matrix(m))
                } else {
                    Err("Matrix has no inverse".to_string())
                }
            }
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Result<Self, String>;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a * b)),
            (Value::Scalar(a), Value::Tuple(b)) => {
                Ok(Value::Tuple(b.into_iter().map(|x| a.clone() * x).collect()))
            }
            (Value::Scalar(a), Value::Matrix(b)) => Ok(Value::Matrix(b * a)),
            (Value::Tuple(a), Value::Scalar(b)) => {
                Ok(Value::Tuple(a.into_iter().map(|x| x * b.clone()).collect()))
            }
            (Value::Matrix(a), Value::Scalar(b)) => Ok(Value::Matrix(a * b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a * b).map(Value::Matrix).map_err(|e| e.to_string())
            }
            (Value::Tuple(_), Value::Tuple(_)) => Err("No multiplication of tuples".to_string()),
            (Value::Tuple(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Tuple(_)) => {
                Err("No multiplication of tuples and matrices".to_string())
            }
        }
    }
}

impl ops::Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Value::Scalar(x) => x.neg().into(),
            Value::Tuple(x) => x.into_iter().map(|x| x.neg()).collect::<Vec<_>>().into(),
            Value::Matrix(x) => x.neg().into(),
        }
    }
}

impl Pow<Value> for Value {
    type Output = Result<Self, String>;

    fn pow(self, other: Self) -> Self::Output {
        match (self.clone(), other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::from(a.pow(b))),
            (Value::Matrix(m), Value::Scalar(b)) => {
                if !m.is_square() {
                    Err("Matrix is not square".to_string())
                } else if b.is_zero() {
                    Ok(Matrix::one(m.rows()).map_err(|e| e.to_string())?.into())
                } else if m.is_diagonal() {
                    let mut acc = Matrix::zero(m.rows(), m.cols()).map_err(|e| e.to_string())?;
                    for i in 0..m.rows() {
                        acc[i][i] = m[i][i].clone().pow(b.clone());
                    }
                    Ok(acc.into())
                } else {
                    match b {
                        Scalar::Int(ref x) => {
                            let (mut num, den) = x.clone().into_numer_denom();
                            if den == 1 {
                                let m = if num < 0 {
                                    num = -num;
                                    self.inv()?
                                } else {
                                    self
                                };
                                let mut acc = m.clone();
                                while num > 1 {
                                    acc = (acc * m.clone())?;
                                    num -= 1;
                                }
                                Ok(acc)
                            } else {
                                // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
                                // https://en.wikipedia.org/wiki/Square_root_of_a_matrix
                                Err(format!("NYI (matrix ^ {num}/{den})"))
                            }
                        }
                        _ => Err("NYI: Matrix raised to non-whole power".to_string()),
                    }
                }
            }
            (Value::Scalar(a), Value::Matrix(m)) => {
                if !m.is_square() {
                    Err("Matrix is not square".to_string())
                } else if m.is_diagonal() {
                    let mut acc = Matrix::zero(m.rows(), m.cols()).map_err(|e| e.to_string())?;
                    for i in 0..m.rows() {
                        acc[i][i] = a.clone().pow(m[i][i].clone());
                    }
                    Ok(acc.into())
                } else {
                    // c.f. RpnMatrixExt::try_exp()
                    // @@TODO: General case a^m == U * a^D * U^-1 where D is a diagonal matrix and m == U * D * U^-1
                    Err("NYI: scalar ^ matrix".to_string())
                }
            }
            (Value::Matrix(_), Value::Matrix(_)) => Err("Unsupported: matrix ^ matrix".to_string()),
            (Value::Tuple(_), _) => Err("No multiplication of tuples".to_string()),
            (_, Value::Tuple(_)) => Err("Illegal Operation: x ^ ( tuple )".to_string()),
        }
    }
}

impl ops::Rem for Value {
    type Output = Result<Self, String>;

    fn rem(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a % b)),
            (Value::Matrix(_a), Value::Scalar(_b)) => Err("NYI".to_string()),
            (Value::Matrix(_a), Value::Matrix(_b)) => Err("NYI".to_string()),
            (Value::Tuple(_), Value::Scalar(_)) | (Value::Scalar(_), Value::Tuple(_)) => {
                Err("NYI: tuple/scalar arithmetic".to_string())
            }
            (Value::Tuple(_), Value::Tuple(_)) => {
                Err("Illegal Operation: tuple/tuple arithmetic".to_string())
            }
            (Value::Tuple(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Tuple(_)) => {
                Err("Illegal Operation: tuple/matrix arithmetic".to_string())
            }
            (Value::Scalar(_), Value::Matrix(_)) => {
                Err("Illegal Operation: matrix/scalar arithmetic".to_string())
            }
        }
    }
}

impl ops::Shl<Value> for Value {
    type Output = Result<Self, String>;

    fn shl(self, other: Self) -> Self::Output {
        match (self, other.get_usize()) {
            (Value::Scalar(Scalar::Int(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            (Value::Scalar(Scalar::Float(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            (a, _) => Err(format!("{a:?} << {other:?} is not REAL << INTEGER(u32)")),
        }
    }
}

impl ops::Shr<Value> for Value {
    type Output = Result<Self, String>;

    fn shr(self, other: Self) -> Self::Output {
        match (self, other.get_usize()) {
            (Value::Scalar(Scalar::Int(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            (Value::Scalar(Scalar::Float(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            (a, _) => Err(format!("{a:?} >> {other:?} is not REAL >> INTEGER(u32)")),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = Result<Self, String>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a - b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a - b).map(Value::Matrix).map_err(|e| e.to_string())
            }
            (Value::Tuple(a), Value::Tuple(b)) => Ok(a
                .into_iter()
                .zip_longest(b)
                .map(|x| match x {
                    EitherOrBoth::Both(x, y) => x - y,
                    EitherOrBoth::Left(x) => x,
                    EitherOrBoth::Right(x) => -x,
                })
                .collect::<Vec<_>>()
                .into()),
            (Value::Tuple(_), Value::Scalar(_)) | (Value::Scalar(_), Value::Tuple(_)) => {
                Err("NYI: tuple/scalar arithmetic".to_string())
            }
            (Value::Tuple(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Tuple(_)) => {
                Err("Illegal Operation: tuple/matrix arithmetic".to_string())
            }
            (Value::Matrix(_), Value::Scalar(_)) | (Value::Scalar(_), Value::Matrix(_)) => {
                Err("Illegal Operation: matrix/scalar arithmetic".to_string())
            }
        }
    }
}

impl ops::Add<Scalar> for Scalar {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a + b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a + b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a + b),
        }
    }
}

impl ops::AddAssign<Scalar> for Scalar {
    fn add_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a + b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a + b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a + b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a + b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a + b),
        };
    }
}

impl ops::BitAnd<Scalar> for Scalar {
    type Output = Result<Scalar, String>;

    fn bitand(self, other: Self) -> Self::Output {
        if let (Scalar::Int(a), Scalar::Int(b)) = (self, other)
            && a.is_integer()
            && b.is_integer()
        {
            Ok(Scalar::from(a.numer().clone() & b.numer().clone()))
        } else {
            Err("Not supported".to_string())
        }
    }
}

impl ops::BitOr<Scalar> for Scalar {
    type Output = Result<Scalar, String>;

    fn bitor(self, other: Self) -> Self::Output {
        if let (Scalar::Int(a), Scalar::Int(b)) = (self, other)
            && a.is_integer()
            && b.is_integer()
        {
            Ok(Scalar::from(a.numer().clone() | b.numer().clone()))
        } else {
            Err("Not supported".to_string())
        }
    }
}

impl ops::BitXor<Scalar> for Scalar {
    type Output = Result<Scalar, String>;

    fn bitxor(self, other: Self) -> Self::Output {
        if let (Scalar::Int(a), Scalar::Int(b)) = (self, other)
            && a.is_integer()
            && b.is_integer()
        {
            Ok(Scalar::from(a.numer().clone() ^ b.numer().clone()))
        } else {
            Err("Not supported".to_string())
        }
    }
}

impl ops::Div<Scalar> for Scalar {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            return Scalar::Float(Float::with_val(1, Special::Nan));
        }
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a / b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a / b),
            (Scalar::Int(a), Scalar::Complex(b)) => {
                let fa = Float::with_val(FLOAT_PRECISION, a);
                Scalar::from(fa / b)
            }

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a / b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a / b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a / b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a / b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a / b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a / b),
        }
    }
}

impl ops::DivAssign<Scalar> for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self = if rhs.is_zero() {
            Scalar::Float(Float::with_val(1, Special::Nan))
        } else {
            match (self.clone(), rhs) {
                (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a / b),
                (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a / b),
                (Scalar::Int(a), Scalar::Complex(b)) => {
                    let fa = Float::with_val(FLOAT_PRECISION, a);
                    Scalar::from(fa / b)
                }

                (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a / b),
                (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a / b),
                (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a / b),

                (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a / b),
                (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a / b),
                (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a / b),
            }
        };
    }
}

impl Inv for Scalar {
    type Output = Scalar;

    fn inv(self) -> Self::Output {
        match self {
            Scalar::Int(x) => Scalar::from(Rational::from((x.denom(), x.numer()))),
            Scalar::Float(x) => Scalar::from(x.recip()),
            Scalar::Complex(x) => Scalar::from(x.recip()),
        }
    }
}

impl ops::Mul<Scalar> for Scalar {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(b * a),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a * b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a * b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a * b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a * b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a * b),
        }
    }
}

impl ops::MulAssign<Scalar> for Scalar {
    fn mul_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(b * a),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a * b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a * b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a * b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a * b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a * b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a * b),
        };
    }
}

impl ops::Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Scalar::Int(x) => x.neg().into(),
            Scalar::Float(x) => x.neg().into(),
            Scalar::Complex(x) => x.neg().into(),
        }
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => a == b,
            (Scalar::Int(a), Scalar::Float(b)) => a == b,
            (Scalar::Int(a), Scalar::Complex(b)) => a == b,

            (Scalar::Float(a), Scalar::Int(b)) => a == b,
            (Scalar::Float(a), Scalar::Float(b)) => a == b,
            (Scalar::Float(a), Scalar::Complex(b)) => a == b,

            (Scalar::Complex(a), Scalar::Int(b)) => a == b,
            (Scalar::Complex(a), Scalar::Float(b)) => a == b,
            (Scalar::Complex(a), Scalar::Complex(b)) => a == b,
        }
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => a.partial_cmp(b),
            (Scalar::Int(a), Scalar::Float(b)) => a.partial_cmp(b),
            (Scalar::Int(a), Scalar::Complex(b)) => (a == b).then_some(Ordering::Equal),

            (Scalar::Float(a), Scalar::Int(b)) => a.partial_cmp(b),
            (Scalar::Float(a), Scalar::Float(b)) => a.partial_cmp(b),
            (Scalar::Float(a), Scalar::Complex(b)) => (a == b).then_some(Ordering::Equal),

            (Scalar::Complex(a), Scalar::Int(b)) => (a == b).then_some(Ordering::Equal),
            (Scalar::Complex(a), Scalar::Float(b)) => (a == b).then_some(Ordering::Equal),
            (Scalar::Complex(a), Scalar::Complex(b)) => (a == b).then_some(Ordering::Equal),
        }
    }
}

impl Pow<Scalar> for Scalar {
    type Output = Self;

    fn pow(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => {
                if b.is_integer() {
                    if let Some(b) = b.numer().to_u32() {
                        Scalar::from(a.pow(b))
                    } else {
                        Scalar::from(
                            Complex::with_val(FLOAT_PRECISION, a)
                                .pow(Complex::with_val(FLOAT_PRECISION, b)),
                        )
                    }
                } else if b == Rational::from((1, 2)) {
                    Scalar::from(Complex::with_val(FLOAT_PRECISION, a).sqrt())
                } else if b == Rational::from((1, 3)) {
                    Scalar::from(Float::with_val(FLOAT_PRECISION, a).cbrt())
                } else {
                    Scalar::from(
                        Complex::with_val(FLOAT_PRECISION, a)
                            .pow(Complex::with_val(FLOAT_PRECISION, b)),
                    )
                }
            }
            (Scalar::Int(a), Scalar::Float(b)) => {
                Scalar::from(Complex::with_val(FLOAT_PRECISION, a).pow(Complex::from(b)))
            }
            (Scalar::Int(a), Scalar::Complex(b)) => {
                Scalar::from(Complex::with_val(FLOAT_PRECISION, a).pow(b))
            }

            (Scalar::Float(a), Scalar::Int(b)) => {
                Scalar::from(Complex::from(a).pow(Float::with_val(FLOAT_PRECISION, b)))
            }
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(Complex::from(a).pow(b)),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(Complex::from(a).pow(b)),

            (Scalar::Complex(a), Scalar::Int(b)) => {
                Scalar::from(a.pow(Float::with_val(FLOAT_PRECISION, b)))
            }
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a.pow(b)),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a.pow(b)),
        }
    }
}

impl ops::Rem for Scalar {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        if other.is_zero() {
            // @@ Infinity vs NegInfinity
            Scalar::Float(Float::with_val(1, Special::Infinity))
        } else {
            match (self, other) {
                (Scalar::Int(a), Scalar::Int(b)) => {
                    if a.is_integer() && b.is_integer() {
                        let (ia, _) = a.into_numer_denom();
                        let ib = b.numer();
                        Scalar::from(ia % ib)
                    } else {
                        let fa: Float = Float::with_val(FLOAT_PRECISION, a);
                        let fb: Float = Float::with_val(FLOAT_PRECISION, b);
                        Scalar::from(fa % fb)
                    }
                }
                (Scalar::Int(a), Scalar::Float(b)) => {
                    let fa: Float = Float::with_val(FLOAT_PRECISION, a);
                    Scalar::from(fa % b)
                }
                (Scalar::Int(a), Scalar::Complex(b)) => {
                    let fa = Float::with_val(FLOAT_PRECISION, &a);
                    let mut c = b.clone();
                    c.div_from_round(fa, (Round::Zero, Round::Zero));
                    Scalar::from(a - c * b)
                }

                (Scalar::Float(a), Scalar::Int(b)) => {
                    let fb = Float::with_val(FLOAT_PRECISION, b);
                    Scalar::from(a % fb)
                }
                (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a % b),
                (Scalar::Float(a), Scalar::Complex(b)) => {
                    let mut c = b.clone();
                    c.div_from_round(a.clone(), (Round::Zero, Round::Zero));
                    Scalar::from(a - c * b)
                }

                (Scalar::Complex(a), Scalar::Int(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scalar::from(a - c * b)
                }
                (Scalar::Complex(a), Scalar::Float(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scalar::from(a - c * b)
                }
                (Scalar::Complex(a), Scalar::Complex(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scalar::from(a - c * b)
                }
            }
        }
    }
}

impl Signed for Scalar {
    fn abs(&self) -> Self {
        match self {
            Scalar::Int(x) => x.clone().abs().into(),
            Scalar::Float(x) => x.clone().abs().into(),
            Scalar::Complex(x) => x.clone().abs().into(),
        }
    }
    fn abs_sub(&self, other: &Self) -> Self {
        if self < other {
            0.into()
        } else {
            self.clone() - other.clone()
        }
    }
    fn signum(&self) -> Self {
        match self {
            Scalar::Int(x) => x.clone().signum().into(),
            Scalar::Float(x) => x.clone().signum().into(),
            Scalar::Complex(x) => Scalar::from(Complex::with_val(
                1,
                (x.real().clone().signum(), x.imag().clone().signum()),
            )),
        }
    }
    fn is_positive(&self) -> bool {
        match self {
            Scalar::Int(x) => x.clone().signum() == 1,
            Scalar::Float(x) => x.is_sign_positive(),
            Scalar::Complex(x) => x.real().is_sign_positive(),
        }
    }
    fn is_negative(&self) -> bool {
        match self {
            Scalar::Int(x) => x.clone().signum() == -1,
            Scalar::Float(x) => x.is_sign_negative(),
            Scalar::Complex(x) => x.real().is_sign_negative(),
        }
    }
}

impl ops::Sub<Scalar> for Scalar {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a - b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a - b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a - b),
        }
    }
}

impl ops::SubAssign for Scalar {
    fn sub_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scalar::Int(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Int(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Int(a), Scalar::Complex(b)) => Scalar::from(a - b),

            (Scalar::Float(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Float(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Float(a), Scalar::Complex(b)) => Scalar::from(a - b),

            (Scalar::Complex(a), Scalar::Int(b)) => Scalar::from(a - b),
            (Scalar::Complex(a), Scalar::Float(b)) => Scalar::from(a - b),
            (Scalar::Complex(a), Scalar::Complex(b)) => Scalar::from(a - b),
        }
    }
}

impl std::iter::Sum for Scalar {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Scalar>,
    {
        let mut a = Scalar::zero();

        for b in iter {
            a += b;
        }
        a
    }
}

impl std::iter::Product for Scalar {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Scalar>,
    {
        let mut a = Scalar::one();

        for b in iter {
            a *= b;
        }
        a
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar::zero()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use libmat::mat::Matrix;
    use num_traits::{One, Zero};
    use std::collections::VecDeque;

    #[test]
    fn scalar_one_equality() {
        let a = Scalar::one();
        let b = Scalar::Float(Float::with_val(FLOAT_PRECISION, 1.0));
        let c = Scalar::Complex(Complex::with_val(FLOAT_PRECISION, (1.0, 0)));

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
        assert!(b.is_one());
        assert!(c.is_one());
    }

    #[test]
    fn scalar_zero_equality() {
        let a: Value = Scalar::zero().into();
        let b: Value = Float::with_val(FLOAT_PRECISION, 0.0).into();
        let c: Value = Complex::with_val(FLOAT_PRECISION, (0.0, 0.0)).into();
        let d: Value = Integer::from(0).into();
        let m: Value = Matrix::<Scalar>::zero(3, 3).unwrap().into();

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(a, d);
        assert_eq!(b, c);
        assert_eq!(b, d);
        assert_eq!(c, d);
        assert!(a.is_zero());
        assert!(b.is_zero());
        assert!(c.is_zero());
        assert!(d.is_zero());
        assert!(m.is_zero());
    }

    #[test]
    fn scalar_factor() {
        let a = Scalar::from(12);
        let f2 = Scalar::from(2);
        let f3 = Scalar::from(3);
        let one = Scalar::from(1);

        assert_eq!(a.factor(), Ok(vec![f2.clone(), f2, f3]));
        assert_eq!(one.clone().factor(), Ok(vec![one]));
    }

    #[test]
    fn value_get_usize_some() {
        let a = Value::from(Scalar::from(rug::Rational::from((5, 1))));
        assert_eq!(a.get_usize(), Some(5));
    }

    #[test]
    fn value_get_usize_none() {
        let b = Value::from(Scalar::from(rug::Rational::from((5, 2))));
        assert_eq!(b.get_usize(), None);
    }

    #[test]
    fn value_matrix_from_str() {
        let b = Value::try_from("[ 1 0 0; 0 1 0; 0 0 1]").unwrap();
        let i3 = Value::identity(3.into()).unwrap();
        assert_eq!(i3, b);
        assert_eq!(Ok(i3), Value::identity(b));
    }

    #[test]
    fn value_matrix_from_str_imaginary() {
        let b = Value::try_from("[ i 0 0; 0 i 0; 0 0 i]").unwrap();
        let i3 = Value::identity(3.into()).unwrap();
        let i = Value::try_from("i").unwrap();

        assert_eq!(i3 * i, Ok(b));
    }

    #[test]
    fn value_tuple_from_str() {
        let a = Value::try_from("( 1 2/3 3 4+2i 5").unwrap();
        let v = Value::Tuple(VecDeque::from([
            1.into(),
            rug::Rational::from((2, 3)).into(),
            3.into(),
            rug::Complex::with_val(FLOAT_PRECISION, (4.0, 2.0)).into(),
            5.into(),
        ]));

        assert_eq!(a, v);
    }

    #[test]
    fn scalar_from_str_complex() {
        let a = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, 2.0)));
        let b = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -2.0)));
        let c = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -1.0)));

        assert_eq!(Scalar::try_from("0+2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 +2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 + 2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0b0.0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("2i"), Ok(a));
        assert_eq!(Scalar::try_from("0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("0 -2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("0 - 2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("-0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("+0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("+0o0-0d2.0i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("-2i"), Ok(b));
        assert_eq!(Scalar::try_from("0 - 1i"), Ok(c.clone()));
        assert_eq!(Scalar::try_from("-1i"), Ok(c.clone()));
        assert_eq!(Scalar::try_from("0-i"), Ok(c.clone()));
        assert_eq!(Scalar::try_from("-i"), Ok(c));
    }

    #[test]
    fn scalar_from_str_rational() {
        let a = Scalar::from(rug::Rational::from((1, 2)));

        assert_eq!(Scalar::try_from("0"), Ok(Scalar::zero()));
        assert_eq!(Scalar::try_from("-1"), Ok(Scalar::from(-1)));
        assert_eq!(Scalar::try_from("+1"), Ok(Scalar::one()));
        assert_eq!(Scalar::try_from("1/2"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0x1/0x2"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0b1/2"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0b1/0o2"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0z1/0z2"), Ok(a));
    }

    #[test]
    fn scalar_to_string_radix() {
        let a = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (10.0, 20.0)));
        let b = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -21.0)));
        let c = Scalar::try_from("1/2").unwrap();

        assert_eq!(
            a.to_string_radix(Radix::Decimal, true, Some(100)).as_str(),
            "10+20i"
        );
        assert_eq!(
            b.to_string_radix(Radix::Decimal, true, Some(100)).as_str(),
            "-21i"
        );
        assert_eq!(
            c.to_string_radix(Radix::Decimal, true, Some(100)).as_str(),
            "1/2"
        );
        assert_eq!(
            a.to_string_radix(Radix::Decimal, false, Some(100)).as_str(),
            "10+20i"
        );
        assert_eq!(
            b.to_string_radix(Radix::Decimal, false, Some(100)).as_str(),
            "-21i"
        );
        assert_eq!(
            c.to_string_radix(Radix::Decimal, false, Some(100)).as_str(),
            "0.5"
        );
        assert_eq!(
            a.to_string_radix(Radix::Hex, true, Some(100)).as_str(),
            "0xa+0x14i"
        );
        assert_eq!(
            b.to_string_radix(Radix::Hex, true, Some(100)).as_str(),
            "-0x15i"
        );
        assert_eq!(
            c.to_string_radix(Radix::Hex, true, Some(100)).as_str(),
            "0x1/0x2"
        );
        assert_eq!(
            c.to_string_radix(Radix::Hex, false, Some(100)).as_str(),
            "0x0.8"
        );
    }

    #[test]
    fn string_scalar_float() {
        let a = rug::Float::with_val(FLOAT_PRECISION, 1.5);
        let b = rug::Float::with_val(FLOAT_PRECISION, -1.5);
        let c = rug::Float::with_val(FLOAT_PRECISION, 0.25);
        let d = rug::Float::with_val(FLOAT_PRECISION, 16.0);
        assert_eq!(a.to_string_scalar(Radix::Decimal), "1.5");
        assert_eq!(a.to_string_scalar(Radix::Hex), "0x1.8");
        assert_eq!(a.to_string_scalar(Radix::Octal), "0o1.4");
        assert_eq!(a.to_string_scalar(Radix::Binary), "0b1.1");
        assert_eq!(b.to_string_scalar(Radix::Decimal), "-1.5");
        assert_eq!(b.to_string_scalar(Radix::Hex), "-0x1.8");
        assert_eq!(b.to_string_scalar(Radix::Octal), "-0o1.4");
        assert_eq!(b.to_string_scalar(Radix::Binary), "-0b1.1");
        assert_eq!(c.to_string_scalar(Radix::Decimal), "0.25");
        assert_eq!(c.to_string_scalar(Radix::Hex), "0x0.4");
        assert_eq!(c.to_string_scalar(Radix::Octal), "0o0.2");
        assert_eq!(c.to_string_scalar(Radix::Binary), "0b0.01");
        assert_eq!(d.to_string_scalar(Radix::Decimal), "16");
        assert_eq!(d.to_string_scalar(Radix::Hex), "0x10");
        assert_eq!(d.to_string_scalar(Radix::Octal), "0o20");
        assert_eq!(d.to_string_scalar(Radix::Binary), "0b10000");
        assert_eq!(d.to_string_scalar(Radix::Duodecimal), "0z14");
    }

    #[test]
    fn string_scalar_rational() {
        let a = rug::Rational::from((-10, 1));
        let b = rug::Rational::from((10, 11));
        let z = rug::Rational::from((0, 1));
        assert_eq!(a.to_string_scalar(Radix::Decimal), "-10");
        assert_eq!(a.to_string_scalar(Radix::Hex), "-0xa");
        assert_eq!(a.to_string_scalar(Radix::Octal), "-0o12");
        assert_eq!(a.to_string_scalar(Radix::Binary), "-0b1010");
        assert_eq!(b.to_string_scalar(Radix::Decimal), "10/11");
        assert_eq!(b.to_string_scalar(Radix::Hex), "0xa/0xb");
        assert_eq!(b.to_string_scalar(Radix::Octal), "0o12/0o13");
        assert_eq!(b.to_string_scalar(Radix::Binary), "0b1010/0b1011");
        assert_eq!(z.to_string_scalar(Radix::Decimal), "0");
        assert_eq!(z.to_string_scalar(Radix::Hex), "0");
        assert_eq!(z.to_string_scalar(Radix::Octal), "0");
        assert_eq!(z.to_string_scalar(Radix::Binary), "0");
    }

    #[test]
    fn matrix_is_diag() {
        let a: Matrix<Scalar> = Matrix::from_vec(
            3,
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
                .into_iter()
                .map(Scalar::from)
                .collect(),
        )
        .unwrap();
        let id3: Matrix<Scalar> = Matrix::one(3).unwrap();

        assert!(id3.is_diagonal());
        assert!(!a.is_diagonal());
    }

    #[test]
    fn value_lines() {
        let a: Value = Matrix::from_vec(
            3,
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
                .into_iter()
                .map(Scalar::from)
                .collect(),
        )
        .unwrap()
        .into();
        let one: Value = 1.into();

        assert_eq!(a.lines(), 3);
        assert_eq!(one.lines(), 1);
    }

    #[test]
    fn value_matrix_scaler_multiplication() {
        let a: Value = Matrix::from_vec(
            3,
            3,
            [1, 2, 3, 1, 2, 3, 1, 2, 3]
                .into_iter()
                .map(Scalar::from)
                .collect(),
        )
        .unwrap()
        .into();
        let b: Value = Matrix::from_vec(
            3,
            3,
            [2, 4, 6, 2, 4, 6, 2, 4, 6]
                .into_iter()
                .map(Scalar::from)
                .collect(),
        )
        .unwrap()
        .into();
        let one: Value = 1.into();
        let two: Value = 2.into();

        assert_eq!(one * a.clone(), Ok(a.clone()));
        assert_eq!(a * two, Ok(b));
    }

    #[test]
    fn value_modulo() {
        let m: Value = Matrix::from_vec(
            3,
            3,
            [1, 2, 3, 1, 2, 3, 1, 2, 3]
                .into_iter()
                .map(Scalar::from)
                .collect(),
        )
        .unwrap()
        .into();

        let ap: Value = 30.into();
        let bp: Value = 40.into();
        let cp: Value = 10.into();
        let an: Value = (-30).into();

        assert!(m.try_modulo(&ap).is_err());
        assert!(ap.try_modulo(&m).is_err());
        assert!(ap.try_modulo(&an).is_err());
        assert_eq!(ap.try_modulo(&bp), Ok(ap.clone()));
        assert_eq!(bp.try_modulo(&ap), Ok(cp.clone()));
        assert_eq!(an.try_modulo(&bp), Ok(cp));
    }

    #[test]
    fn value_matrix_stats() {
        let b = Value::try_from("[ -99 -10+4i -1; 6 -14 5; 10-4i 11 101]").unwrap();

        assert_eq!(b.clone().median(), 5.into());
        assert_eq!(b.clone().mean(), 1.into());
        assert_eq!(b.sum(), 9.into());
    }

    #[test]
    fn value_sigma() {
        let b = Value::try_from("(2 4 4 4 5 5 7 9)").unwrap();

        assert_eq!(b.clone().mean(), 5.into());
        assert_eq!(b.standard_deviation(), 2.into());
    }

    #[test]
    fn value_matrix_roundings() {
        let a = Value::try_from("[ -4.5 2-5.1i 3/2 ]").unwrap();

        assert_eq!(a.clone().floor(), Value::try_from("[ -5 2-6i 1 ]").unwrap());
        assert_eq!(a.clone().ceil(), Value::try_from("[ -4 2-5i 2 ]").unwrap());
        assert_eq!(a.clone().round(), Value::try_from("[ -5 2-5i 2 ]").unwrap());
        assert_eq!(a.trunc(), Value::try_from("[ -4 2-5i 1 ]").unwrap());
    }

    #[test]
    fn value_tuple_roundings() {
        let a = Value::try_from("( -4.5 2-5.1i 3/2 )").unwrap();

        assert_eq!(a.clone().floor(), Value::try_from("( -5 2-6i 1 )").unwrap());
        assert_eq!(a.clone().ceil(), Value::try_from("( -4 2-5i 2 )").unwrap());
        assert_eq!(a.clone().round(), Value::try_from("( -5 2-5i 2 )").unwrap());
        assert_eq!(a.trunc(), Value::try_from("( -4 2-5i 1 )").unwrap());
    }

    #[test]
    fn to_string_scalar() {
        let a = Rational::from((4, 3));

        assert_eq!(a.digits(Radix::default()), 3);
        assert_eq!(a.digits(Radix::Hex), 7);
    }
}
