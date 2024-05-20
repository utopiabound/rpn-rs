/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use libmat::{mat::Matrix, matrix};
use num_traits::{Inv, Num, One, Signed, Zero};
use regex::Regex;
use rug::{
    float::{Constant, Round, Special},
    ops::{DivAssignRound, DivFromRound, Pow, RemRounding},
    Complete, Complex, Float, Integer, Rational,
};
use std::{
    cmp::{max, min, Ordering},
    ops,
};

const FLOAT_PRECISION: u32 = 256;

#[derive(Debug, Clone)]
pub enum Scalar {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(Scalar),
    Matrix(Matrix<Scalar>),
}

impl From<Matrix<Scalar>> for Value {
    fn from(x: Matrix<Scalar>) -> Self {
        Value::Matrix(x)
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
                    s + &"0".repeat((exp - len) as usize)
                } else {
                    let mut v = s;
                    v.insert(exp as usize, '.');
                    v.trim_end_matches('0').trim_end_matches('.').to_string()
                }
            } else {
                "0.".to_string() + &"0".repeat(exp.abs_diff(0) as usize) + s.trim_end_matches('0')
            }
        } else if self.is_zero() {
            s
        } else {
            return s;
        };
        format!("{}{}{s}", if sign { "-" } else { "" }, radix.prefix())
    }
    fn digits(&self, radix: Radix) -> usize {
        let (_sign, s, _exp) = self.to_sign_string_exp(radix.into(), None);
        s.len() // @@ FIXME: large negative exp
    }
    fn to_string_width(&self, radix: Radix, width: Option<usize>) -> String {
        let width = width.map(|w| {
            let l = self.digits(radix);
            let w = w - 1 - radix.prefix().len() - if self.is_sign_negative() { 1 } else { 0 };
            min(w, l)
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

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum Radix {
    #[default]
    Decimal,
    Hex,
    Binary,
    Octal,
}

impl Radix {
    pub fn prefix(&self) -> &str {
        match self {
            Radix::Decimal => "",
            Radix::Hex => "0x",
            Radix::Binary => "0b",
            Radix::Octal => "0o",
        }
    }
    pub fn from_prefix(val: &str) -> Self {
        match val.to_lowercase().as_str() {
            "0b" | "b" => Self::Binary,
            "0o" | "o" => Self::Octal,
            "0d" | "d" => Self::Decimal,
            "0x" | "x" => Self::Hex,
            _ => Self::default(),
        }
    }

    // (is_neg, Radix, rest_of_string)
    pub fn split_string(value: &str) -> (bool, Self, String) {
        let re = Regex::new(r"^(-)?(0[xXoObBdD])?(.*)").unwrap();
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
        }
    }
}

impl From<i32> for Scalar {
    fn from(x: i32) -> Self {
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
        if x.is_integer() {
            Scalar::Int(Rational::from(x.to_integer().unwrap()))
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
    let (f, r, v) = Radix::split_string(val);
    let v = Float::parse_radix(format!("{}{v}", if f { "-" } else { "" }), r.into())
        .map_err(|e| format!("{v}: {e}"))?;
    Ok(Float::with_val(FLOAT_PRECISION, v))
}

fn parse_int(val: &str) -> Result<Integer, String> {
    let (f, r, v) = Radix::split_string(val);
    let v = Integer::parse_radix(format!("{}{v}", if f { "-" } else { "" }), r.into())
        .map_err(|e| format!("{v}: {e}"))?;
    Ok(v.complete())
}

impl TryFrom<&str> for Scalar {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.contains(&['(', 'i'][..]) {
            // Change "1+2i" -> "(1 2)"
            let re = Regex::new(
                r"(\(?(?P<sr>[+-])?\s*(?P<r>(?:0[xXoObBdD])?[[:xdigit:]]+(?:[.][[:xdigit:]]+)?))?\s*(?P<si>[+-])?\s*(?P<i>(?:0[xXoObBdD])?[[:xdigit:]]+(?:[.][[:xdigit:]]+)?)[i)]",
            )
                .unwrap();
            let caps = re
                .captures(value)
                .ok_or_else(|| format!("Failed to parse: {value}"))?;
            let real = format!(
                "{}{}",
                caps.name("sr").map(|x| x.as_str()).unwrap_or(""),
                caps.name("r").map(|x| x.as_str()).unwrap_or("0")
            );
            let imag = format!(
                "{}{}",
                caps.name("si").map(|x| x.as_str()).unwrap_or(""),
                caps.name("i").map(|x| x.as_str()).unwrap_or("0")
            );

            let r = parse_float(&real)?;
            let i = parse_float(&imag)?;

            Ok(Scalar::from(Complex::from((r, i))))
        } else if value.contains('[') {
            Err("Parsing Matrix as Scalar".to_string())
        } else if value.contains('.') {
            Ok(Scalar::from(parse_float(value)?))
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
    pub fn exp(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).exp()),
            Scalar::Float(x) => Scalar::from(x.exp()),
            Scalar::Complex(x) => Scalar::from(x.exp()),
        }
    }

    /// Natural Logarithm
    pub fn ln(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).ln()),
            Scalar::Float(x) => Scalar::from(x.ln()),
            Scalar::Complex(x) => Scalar::from(x.ln()),
        }
    }

    /// Logarithm (base 10)
    pub fn log10(self) -> Self {
        match self {
            Scalar::Int(x) => Scalar::from(Float::with_val(FLOAT_PRECISION, x).log10()),
            Scalar::Float(x) => Scalar::from(x.log10()),
            Scalar::Complex(x) => Scalar::from(x.log10()),
        }
    }

    /// Logarithm (base 2)
    pub fn try_log2(self) -> Result<Self, String> {
        match self {
            Scalar::Int(x) => Ok(Scalar::from(Float::with_val(FLOAT_PRECISION, x).log2())),
            Scalar::Float(x) => Ok(Scalar::from(x.log2())),
            // mpc is missing a log2() function
            Scalar::Complex(_) => Err("Log2() not available for Complex Numbers".to_string()),
        }
    }

    /// Factorial (`n!`)
    pub fn try_factorial(self) -> Result<Self, String> {
        match self {
            Scalar::Int(x) if x.is_integer() => x.numer().factorial().map(|x| x.into()),
            _ => Err("n! only implemented for integers".to_string()),
        }
    }

    /// Permutation (`nPm`)
    /// `P(n, m) = n! / (n - m)!`
    pub fn try_permute(self, r: Self) -> Result<Self, String> {
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
    pub fn try_choose(self, r: Self) -> Result<Self, String> {
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
    pub fn factor(self) -> Result<Vec<Self>, String> {
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
            Scalar::Complex(_) => Err("NYI".to_string()), // @@
        }
    }

    /// Truncate to Integer
    pub fn trunc(&self) -> Result<Self, String> {
        match self {
            Scalar::Int(x) => Ok(x.clone().trunc().into()),
            Scalar::Float(x) => Ok(x.clone().trunc().into()),
            Scalar::Complex(_x) => Err("No Truncation of Complex Values".to_string()),
        }
    }

    /// Round to nearest Integer
    pub fn round(&self) -> Result<Self, String> {
        match self {
            Scalar::Int(x) => Ok(x.clone().round().into()),
            Scalar::Float(x) => Ok(x.clone().round().into()),
            Scalar::Complex(_x) => Err("No Rounding of Complex Values".to_string()),
        }
    }

    /// Return the numerator (as `usize`) iff denominator is 1
    fn get_usize(self) -> Option<usize> {
        match self {
            Scalar::Int(x) => x.is_integer().then(|| x.numer().to_usize()).flatten(),
            _ => None,
        }
    }

    pub fn to_string_radix(&self, radix: Radix, rational: bool, width: Option<usize>) -> String {
        //eprintln!("{self:?} digits:{digits:?}");
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
                    let digits = width - sign.len() - 1 - radix.prefix().len() * 2;
                    let (rlen, ilen) = if digits > rlen + ilen {
                        (rlen, ilen)
                    } else {
                        let t = rlen + ilen;
                        //eprintln!("digits:{digits} t:{t} :: {r} ({rlen}) + {i} ({ilen}) i");
                        (max(2, rlen * digits / t) - 1, max(2, ilen * digits / t) - 1)
                    };
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

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Scalar(x) => x.is_zero(),
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

    pub fn try_sqrt(self) -> Result<Self, String> {
        self.try_root(2.into())
    }

    pub fn to_string_radix(
        &self,
        radix: Radix,
        rational: bool,
        flat: bool,
        digits: impl Into<Option<usize>>,
    ) -> String {
        let digits = digits.into();
        match self {
            Value::Scalar(x) => x.to_string_radix(radix, rational, digits),
            Value::Matrix(x) => {
                let mut s = String::from("[");
                let rowmax = x.rows();
                // @@ FIXME: digits
                for i in 0..rowmax {
                    for j in 0..x.cols() {
                        s += " ";
                        s += x[i][j].to_string_radix(radix, rational, digits).as_str();
                    }
                    s += if i == rowmax - 1 {
                        " ]"
                    } else if flat {
                        " ;"
                    } else {
                        " ;\n"
                    };
                }
                s
            }
        }
    }

    pub fn lines(&self) -> usize {
        match self {
            Value::Scalar(_) => 1,
            Value::Matrix(x) => x.rows(),
        }
    }

    pub fn try_factor(self) -> Result<Vec<Value>, String> {
        match self {
            Value::Scalar(x) => x
                .factor()
                .map(|x| x.into_iter().map(Value::Scalar).collect()),
            Value::Matrix(_) => Err("No prime factors of matricies".to_string()),
        }
    }

    pub fn try_factorial(self) -> Result<Value, String> {
        match self {
            Value::Scalar(x) => Ok(x.try_factorial()?.into()),
            _ => Err(format!("{self:?}! is not INT!")),
        }
    }

    pub fn try_permute(self, r: Self) -> Result<Value, String> {
        match (self, r) {
            (Value::Scalar(x), Value::Scalar(r)) => Ok(x.try_permute(r)?.into()),
            _ => Err("Permuation only implement for INT P INT".to_string()),
        }
    }

    pub fn try_choose(self, r: Self) -> Result<Value, String> {
        match (self, r) {
            (Value::Scalar(x), Value::Scalar(r)) => Ok(x.try_choose(r)?.into()),
            _ => Err("Combination only implement for INT C INT".to_string()),
        }
    }

    pub fn try_modulo(&self, b: &Value) -> Result<Value, String> {
        if b.is_zero() {
            Err("Division by zero".to_string())
        } else if let (Some(a), Some(b)) = (self.get_integer(), b.get_integer()) {
            Ok(Value::from(a.rem_euc(b)))
        } else {
            Err(format!("{self:?} mod {b:?} is not INT mod INT"))
        }
    }

    pub fn try_root(self, other: Value) -> Result<Self, String> {
        self.pow(other.inv()?)
    }

    pub fn try_rshift(&self, b: &Value) -> Result<Self, String> {
        match (self, b.get_usize()) {
            (Value::Scalar(Scalar::Int(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            (Value::Scalar(Scalar::Float(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            _ => Err(format!("{self:?} >> {b:?} is not REAL >> INTEGER(u32)")),
        }
    }

    pub fn try_lshift(&self, b: &Value) -> Result<Self, String> {
        match (self, b.get_usize()) {
            (Value::Scalar(Scalar::Int(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            (Value::Scalar(Scalar::Float(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            _ => Err(format!("{self:?} << {b:?} is not REAL << INTEGER(u32)")),
        }
    }

    pub fn try_ln(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.ln())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_abs(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.abs())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_exp(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.exp())),
            Value::Matrix(x) => x.try_exp().map(Value::Matrix),
        }
    }

    pub fn try_log10(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.log10())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_log2(self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(Value::Scalar(x.try_log2()?)),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_dms_conv(&self) -> Result<Self, String> {
        match &self {
            Value::Scalar(x) => {
                let mut m = matrix! { Scalar::zero(), Scalar::zero(), Scalar::zero()};
                m[0][0] = x.trunc()? % Scalar::from(360);
                let f = x.abs_sub(&m[0][0]) * Scalar::from(60);
                m[0][1] = f.trunc()?;
                let f = (f - m[0][1].clone()) * Scalar::from(60);
                m[0][2] = f;
                Ok(m.into())
            }
            Value::Matrix(m) => {
                if m.rows() != 1 || m.cols() > 3 {
                    Err(format!(
                        "Matrix of incorrect size [{}x{}] expected [1x3]",
                        m.rows(),
                        m.cols()
                    ))
                } else {
                    Ok(((m[0].first().cloned().unwrap_or_default()
                        + m[0].get(1).cloned().unwrap_or_default() / Scalar::from(60)
                        + m[0].get(2).cloned().unwrap_or_default() / Scalar::from(3600))
                        % Scalar::from(360))
                    .into())
                }
            }
        }
    }

    /// Truncate values to Integer
    pub fn try_trunc(&self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(x.trunc()?.into()),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].trunc()?;
                    }
                }
                Ok(m.into())
            }
        }
    }

    /// Round values to Integer
    pub fn try_round(&self) -> Result<Self, String> {
        match self {
            Value::Scalar(x) => Ok(x.round()?.into()),
            Value::Matrix(x) => {
                let mut m = x.clone();
                for i in 0..x.rows() {
                    for j in 0..x.cols() {
                        m[i][j] = x[i][j].round()?;
                    }
                }
                Ok(m.into())
            }
        }
    }

    // Matrix Only Functions
    pub fn try_det(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No determinate for Scalar".to_string()),
            Value::Matrix(x) => Ok(Value::Scalar(x.det().map_err(|e| e.to_string())?)),
        }
    }

    pub fn try_rref(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No determinate for Scalar".to_string()),
            Value::Matrix(x) => Ok(Value::Matrix(x.rref())),
        }
    }

    pub fn try_transpose(self) -> Result<Self, String> {
        match self {
            Value::Scalar(_) => Err("No Transpose for Scalar".to_string()),
            Value::Matrix(x) => Ok(Value::Matrix(x.transpose())),
        }
    }

    // Constants
    pub fn identity(n: Value) -> Result<Self, String> {
        match n {
            Value::Scalar(n) => {
                if let Some(x) = n.get_usize() {
                    Matrix::one(x).map(Value::Matrix).map_err(|e| e.to_string())
                } else {
                    Err("Identity Matrix can only be created with integer size".to_string())
                }
            }
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

    pub fn ones(n: Value) -> Result<Self, String> {
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
            Value::Matrix(x) => Matrix::new(x.rows(), x.cols(), Scalar::one())
                .map(Value::Matrix)
                .map_err(|e| e.to_string()),
        }
    }

    pub fn e() -> Self {
        let f = Float::with_val(FLOAT_PRECISION, 1);
        Value::Scalar(Scalar::from(f.exp()))
    }
    pub fn i() -> Self {
        Value::Scalar(Scalar::from(Complex::with_val(FLOAT_PRECISION, (0, 1))))
    }
    pub fn pi() -> Self {
        Value::Scalar(Scalar::from(Float::with_val(FLOAT_PRECISION, Constant::Pi)))
    }
    pub fn catalan() -> Self {
        Value::Scalar(Scalar::from(Float::with_val(
            FLOAT_PRECISION,
            Constant::Catalan,
        )))
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
            _ => Err("Illegal Operation: Scalar and Matrix Addition".to_string()),
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
            (Value::Scalar(a), Value::Matrix(b)) => Ok(Value::Matrix(b * a)),
            (Value::Matrix(a), Value::Scalar(b)) => Ok(Value::Matrix(a * b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a * b).map(Value::Matrix).map_err(|e| e.to_string())
            }
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
        }
    }
}

impl ops::Rem for Value {
    type Output = Result<Self, String>;

    fn rem(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a % b)),
            (Value::Matrix(_a), Value::Scalar(_b)) => Err("NYI".to_string()),
            _ => Err("Illegal Operation: ".to_string()),
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
            _ => Err("Illegal Operation: Matrix and Scalar Subtraction".to_string()),
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
                    let fa: Float = Float::with_val(FLOAT_PRECISION, a);
                    let fb: Float = Float::with_val(FLOAT_PRECISION, b);
                    Scalar::from(fa % fb)
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
    fn scalar_trunc() {
        let a: Scalar = rug::Float::with_val(FLOAT_PRECISION, 1.5).into();
        let b: Scalar = rug::Float::with_val(FLOAT_PRECISION, -1.5).into();

        assert_eq!(a.trunc(), Ok(Scalar::one()));
        assert_eq!(b.trunc(), Ok(-Scalar::one()));
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
    fn scalar_from_str_complex() {
        let a = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, 2.0)));
        let b = Scalar::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -2.0)));

        assert_eq!(Scalar::try_from("(0 2)"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0+2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 +2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 + 2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("0b0.0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scalar::try_from("2i"), Ok(a));
        assert_eq!(Scalar::try_from("(0 -2)"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("0 -2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("0 - 2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("-0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("+0-2i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("+0o0-0d2.0i"), Ok(b.clone()));
        assert_eq!(Scalar::try_from("-2i"), Ok(b));
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
        assert_eq!(Scalar::try_from("0b1/0o2"), Ok(a));
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
}
