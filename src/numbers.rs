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
use std::{cmp::Ordering, ops};

const FLOAT_PRECISION: u32 = 256;
// This determines the number of digits that Floats are rounded to.
// This allow decimal 0.1 to print as "0.1" instead of "0.10...02"
const FLOAT_STRING_DIGITS: usize = 72;

#[derive(Debug, Clone)]
pub enum Scaler {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scaler(Scaler),
    Matrix(Matrix<Scaler>),
}

impl From<Matrix<Scaler>> for Value {
    fn from(x: Matrix<Scaler>) -> Self {
        Value::Matrix(x)
    }
}

impl<T: Into<Scaler>> From<T> for Value {
    fn from(x: T) -> Self {
        Value::Scaler(x.into())
    }
}

trait RpnMatrixExt {
    fn is_diagonal(&self) -> bool;
    fn try_exp(&self) -> Result<Self, String>
    where
        Self: Sized;
}

impl RpnMatrixExt for Matrix<Scaler> {
    fn is_diagonal(&self) -> bool {
        if self.is_square() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    if i != j && self[i][j] != Scaler::zero() {
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
            (ident + (self.clone() * (Scaler::one().exp() - Scaler::one())))
                .map_err(|e| e.to_string())
        } else {
            // c.f. impl Pow<Value> for Value
            // @@TODO Genral Diagonalizable: https://en.wikipedia.org/wiki/Matrix_exponential#Diagonalizable_case
            Err("NYI: non-trivial exp(M)".to_string())
        }
    }
}

trait RpnToStringScaler {
    fn to_string_scaler(&self, radix: Radix) -> String;
}

impl RpnToStringScaler for Float {
    fn to_string_scaler(&self, radix: Radix) -> String {
        let (sign, s, exp) = self.to_sign_string_exp(radix.into(), Some(FLOAT_STRING_DIGITS));
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
}

impl RpnToStringScaler for Rational {
    fn to_string_scaler(&self, radix: Radix) -> String {
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

impl From<i32> for Scaler {
    fn from(x: i32) -> Self {
        Scaler::Int(Rational::from(x))
    }
}

impl From<Integer> for Scaler {
    fn from(x: Integer) -> Self {
        Scaler::Int(Rational::from(x))
    }
}

impl From<Rational> for Scaler {
    fn from(x: Rational) -> Self {
        Scaler::Int(x)
    }
}

impl From<Float> for Scaler {
    fn from(x: Float) -> Self {
        if x.is_integer() {
            Scaler::Int(Rational::from(x.to_integer().unwrap()))
        } else {
            Scaler::Float(x)
        }
    }
}

impl From<Complex> for Scaler {
    fn from(x: Complex) -> Self {
        if x.imag().is_zero() {
            Scaler::from(x.real().clone())
        } else {
            Scaler::Complex(x)
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

impl TryFrom<&str> for Scaler {
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

            Ok(Scaler::from(Complex::from((r, i))))
        } else if value.contains('[') {
            Err("Parsing Matrix as Scaler".to_string())
        } else if value.contains('.') {
            Ok(Scaler::from(parse_float(value)?))
        } else if let Some((numer, denom)) = value.split_once('/') {
            let n = parse_int(numer)?;
            let d = parse_int(denom)?;

            Ok(Scaler::from(Rational::from((n, d))))
        } else {
            let v = parse_int(value)?;
            Ok(Scaler::from(Rational::from(v)))
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
                    .map(Scaler::try_from)
                    .collect::<Result<Vec<Scaler>, String>>()?,
            )
            .map(|m| m.into())
            .map_err(|e| e.to_string())
        } else {
            Scaler::try_from(value).map(|s| s.into())
        }
    }
}

impl Num for Scaler {
    type FromStrRadixErr = String;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Rational::parse_radix(str, radix as i32)
            .map(|v| Scaler::from(Rational::from(v)))
            .map_err(|e| e.to_string())
    }
}

impl Scaler {
    /// Exponent (e^x)
    pub fn exp(self) -> Self {
        match self {
            Scaler::Int(x) => {
                let f: Float = x * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(f.exp())
            }
            Scaler::Float(x) => Scaler::from(x.exp()),
            Scaler::Complex(x) => Scaler::from(x.exp()),
        }
    }

    /// Natural Logarithm
    pub fn ln(self) -> Self {
        match self {
            Scaler::Int(x) => {
                let f: Float = x * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(f.ln())
            }
            Scaler::Float(x) => Scaler::from(x.ln()),
            Scaler::Complex(x) => Scaler::from(x.ln()),
        }
    }

    /// Logarithm (base 10)
    pub fn log10(self) -> Self {
        match self {
            Scaler::Int(x) => {
                let f: Float = x * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(f.log10())
            }
            Scaler::Float(x) => Scaler::from(x.log10()),
            Scaler::Complex(x) => Scaler::from(x.log10()),
        }
    }

    /// Factor Integer
    pub fn factor(self) -> Result<Vec<Self>, String> {
        match self {
            Scaler::Int(x) => {
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
            Scaler::Float(_) => Err("Cannot Factor Floating point".to_string()),
            Scaler::Complex(_) => Err("NYI".to_string()), // @@
        }
    }

    /// Truncate to Integer
    pub fn trunc(&self) -> Result<Self, String> {
        match self {
            Scaler::Int(x) => Ok(x.clone().trunc().into()),
            Scaler::Float(x) => Ok(x.clone().trunc().into()),
            Scaler::Complex(_x) => Err("No Truncation of Complex Values".to_string()),
        }
    }

    /// Return the numberator (as usize) iff denominator is 1
    fn get_usize(self) -> Option<usize> {
        match self {
            Scaler::Int(x) => x.is_integer().then(|| x.numer().to_usize()).flatten(),
            _ => None,
        }
    }

    pub fn to_string_radix(&self, radix: Radix, rational: bool) -> String {
        match self {
            Scaler::Int(x) => {
                if !rational && !x.is_integer() {
                    let f: Float = x * Float::with_val(32, 1.0);
                    f.to_string_scaler(radix)
                } else {
                    x.to_string_scaler(radix)
                }
            }
            Scaler::Float(x) => x.to_string_scaler(radix),
            Scaler::Complex(x) => {
                let r = x.real();
                let i = x.imag();
                if r.is_zero() {
                    format!("{}i", i.to_string_scaler(radix))
                } else {
                    let sign = if i.is_sign_positive() { "+" } else { "" };
                    format!(
                        "{}{}{}i",
                        r.to_string_scaler(radix),
                        sign,
                        i.to_string_scaler(radix)
                    )
                }
            }
        }
    }
}

impl One for Scaler {
    fn one() -> Self {
        Scaler::Int(Rational::from(1))
    }
}

impl Zero for Scaler {
    fn zero() -> Self {
        Scaler::Int(Rational::from(0))
    }

    /// True if value is Zero
    fn is_zero(&self) -> bool {
        match self {
            Scaler::Int(x) => x.numer().to_u32() == Some(0),
            Scaler::Float(x) => x.is_zero(),
            Scaler::Complex(x) => x.real().is_zero(),
        }
    }
}

impl Value {
    /// Return the numerator iff denominator is 1
    fn get_integer(&self) -> Option<Integer> {
        match self {
            Value::Scaler(Scaler::Int(x)) => x.is_integer().then(|| x.numer().clone()),
            _ => None,
        }
    }

    /// Return the numerator (as usize) iff denominator is 1
    fn get_usize(&self) -> Option<usize> {
        match self {
            Value::Scaler(Scaler::Int(x)) => x.is_integer().then(|| x.numer().to_usize()).flatten(),
            _ => None,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Scaler(x) => x.is_zero(),
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

    pub fn to_string_radix(&self, radix: Radix, rational: bool, flat: bool) -> String {
        match self {
            Value::Scaler(x) => x.to_string_radix(radix, rational),
            Value::Matrix(x) => {
                let mut s = String::from("[");
                let rowmax = x.rows();
                for i in 0..rowmax {
                    for j in 0..x.cols() {
                        s += " ";
                        s += x[i][j].to_string_radix(radix, rational).as_str();
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
            Value::Scaler(_) => 1,
            Value::Matrix(x) => x.rows(),
        }
    }

    pub fn try_factor(self) -> Result<Vec<Value>, String> {
        match self {
            Value::Scaler(x) => x
                .factor()
                .map(|x| x.into_iter().map(Value::Scaler).collect()),
            Value::Matrix(_) => Err("No prime factors of matricies".to_string()),
        }
    }

    pub fn try_factorial(self) -> Result<Value, String> {
        match self {
            Value::Scaler(Scaler::Int(x)) if x.is_integer() && x.clone().signum() >= 0 => {
                let max = x.numer();
                let mut n = Integer::from(1);
                let mut i = Integer::from(2);
                while i <= *max {
                    n *= i.clone();
                    i += 1;
                }
                Ok(n.into())
            }
            _ => Err(format!("{self:?}! is not INT!")),
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
            (Value::Scaler(Scaler::Int(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            (Value::Scaler(Scaler::Float(a)), Some(b)) => Ok(Value::from(a.clone() >> b)),
            _ => Err(format!("{self:?} >> {b:?} is not REAL >> INTEGER(u32)")),
        }
    }

    pub fn try_lshift(&self, b: &Value) -> Result<Self, String> {
        match (self, b.get_usize()) {
            (Value::Scaler(Scaler::Int(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            (Value::Scaler(Scaler::Float(a)), Some(b)) => Ok(Value::from(a.clone() << b)),
            _ => Err(format!("{self:?} << {b:?} is not REAL << INTEGER(u32)")),
        }
    }

    pub fn try_ln(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.ln())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_abs(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.abs())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_exp(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.exp())),
            Value::Matrix(x) => x.try_exp().map(Value::Matrix),
        }
    }

    pub fn try_log10(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.log10())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_dms_conv(&self) -> Result<Self, String> {
        match &self {
            Value::Scaler(x) => {
                let mut m = matrix! { Scaler::zero(), Scaler::zero(), Scaler::zero()};
                m[0][0] = x.trunc()? % Scaler::from(360);
                let f = x.abs_sub(&m[0][0]) * Scaler::from(60);
                m[0][1] = f.trunc()?;
                let f = (f - m[0][1].clone()) * Scaler::from(60);
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
                    Ok(((m[0].get(0).cloned().unwrap_or_default()
                        + m[0].get(1).cloned().unwrap_or_default() / Scaler::from(60)
                        + m[0].get(2).cloned().unwrap_or_default() / Scaler::from(3600))
                        % Scaler::from(360))
                    .into())
                }
            }
        }
    }

    /// Truncate values to Integer
    pub fn try_trunc(&self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(x.trunc()?.into()),
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

    // Matrix Only Functions
    pub fn try_det(self) -> Result<Self, String> {
        match self {
            Value::Scaler(_) => Err("No determinate for Scaler".to_string()),
            Value::Matrix(x) => Ok(Value::Scaler(x.det().map_err(|e| e.to_string())?)),
        }
    }

    pub fn try_rref(self) -> Result<Self, String> {
        match self {
            Value::Scaler(_) => Err("No determinate for Scaler".to_string()),
            Value::Matrix(x) => Ok(Value::Matrix(x.rref())),
        }
    }

    pub fn try_transpose(self) -> Result<Self, String> {
        match self {
            Value::Scaler(_) => Err("No Transpose for Scaler".to_string()),
            Value::Matrix(x) => Ok(Value::Matrix(x.transpose())),
        }
    }

    // Constants
    pub fn identity(n: Value) -> Result<Self, String> {
        match n {
            Value::Scaler(n) => {
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
            Value::Scaler(n) => {
                if let Some(x) = n.get_usize() {
                    Matrix::new(x, x, Scaler::one())
                        .map(Value::Matrix)
                        .map_err(|e| e.to_string())
                } else {
                    Err("Identity Matrix can only be created with integer size".to_string())
                }
            }
            Value::Matrix(x) => Matrix::new(x.rows(), x.cols(), Scaler::one())
                .map(Value::Matrix)
                .map_err(|e| e.to_string()),
        }
    }

    pub fn e() -> Self {
        let f = Float::with_val(FLOAT_PRECISION, 1);
        Value::Scaler(Scaler::from(f.exp()))
    }
    pub fn i() -> Self {
        Value::Scaler(Scaler::from(Complex::with_val(FLOAT_PRECISION, (0, 1))))
    }
    pub fn pi() -> Self {
        Value::Scaler(Scaler::from(Float::with_val(FLOAT_PRECISION, Constant::Pi)))
    }
}

impl ops::Add<Value> for Value {
    type Output = Result<Self, String>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::Scaler(a + b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a + b).map(Value::Matrix).map_err(|e| e.to_string())
            }
            _ => Err("Illegal Operation: Scaler and Matrix Addition".to_string()),
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
            Value::Scaler(x) => {
                if x.is_zero() {
                    Err("Error: Division by zero".to_string())
                } else {
                    Ok(Value::Scaler(x.inv()))
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
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::Scaler(a * b)),
            (Value::Scaler(a), Value::Matrix(b)) => Ok(Value::Matrix(b * a)),
            (Value::Matrix(a), Value::Scaler(b)) => Ok(Value::Matrix(a * b)),
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
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::from(a.pow(b))),
            (Value::Matrix(m), Value::Scaler(b)) => {
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
                        Scaler::Int(ref x) => {
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
            (Value::Scaler(a), Value::Matrix(m)) => {
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
                    Err("NYI: scaler ^ matrix".to_string())
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
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::Scaler(a % b)),
            (Value::Matrix(_a), Value::Scaler(_b)) => Err("NYI".to_string()),
            _ => Err("Illegal Operation: ".to_string()),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = Result<Self, String>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::Scaler(a - b)),
            (Value::Matrix(a), Value::Matrix(b)) => {
                (a - b).map(Value::Matrix).map_err(|e| e.to_string())
            }
            _ => Err("Illegal Operation: Matrix and Scaler Subtraction".to_string()),
        }
    }
}

impl ops::Add<Scaler> for Scaler {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a + b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a + b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a + b),
        }
    }
}

impl ops::AddAssign<Scaler> for Scaler {
    fn add_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a + b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a + b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a + b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a + b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a + b),
        };
    }
}

impl ops::Div<Scaler> for Scaler {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            return Scaler::Float(Float::with_val(1, Special::Nan));
        }
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a / b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a / b),
            (Scaler::Int(a), Scaler::Complex(b)) => {
                let f: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(f / b)
            }

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a / b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a / b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a / b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a / b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a / b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a / b),
        }
    }
}

impl ops::DivAssign<Scaler> for Scaler {
    fn div_assign(&mut self, rhs: Self) {
        *self = if rhs.is_zero() {
            Scaler::Float(Float::with_val(1, Special::Nan))
        } else {
            match (self.clone(), rhs) {
                (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a / b),
                (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a / b),
                (Scaler::Int(a), Scaler::Complex(b)) => {
                    let f: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                    Scaler::from(f / b)
                }

                (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a / b),
                (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a / b),
                (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a / b),

                (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a / b),
                (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a / b),
                (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a / b),
            }
        };
    }
}

impl Inv for Scaler {
    type Output = Scaler;

    fn inv(self) -> Self::Output {
        match self {
            Scaler::Int(x) => Scaler::from(Rational::from((x.denom(), x.numer()))),
            Scaler::Float(x) => Scaler::from(x.recip()),
            Scaler::Complex(x) => Scaler::from(x.recip()),
        }
    }
}

impl ops::Mul<Scaler> for Scaler {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(b * a),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a * b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a * b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a * b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a * b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a * b),
        }
    }
}

impl ops::MulAssign<Scaler> for Scaler {
    fn mul_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(b * a),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a * b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a * b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a * b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a * b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a * b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a * b),
        };
    }
}

impl ops::Neg for Scaler {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Scaler::Int(x) => x.neg().into(),
            Scaler::Float(x) => x.neg().into(),
            Scaler::Complex(x) => x.neg().into(),
        }
    }
}

impl PartialEq for Scaler {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => a == b,
            (Scaler::Int(a), Scaler::Float(b)) => a == b,
            (Scaler::Int(a), Scaler::Complex(b)) => a == b,

            (Scaler::Float(a), Scaler::Int(b)) => a == b,
            (Scaler::Float(a), Scaler::Float(b)) => a == b,
            (Scaler::Float(a), Scaler::Complex(b)) => a == b,

            (Scaler::Complex(a), Scaler::Int(b)) => a == b,
            (Scaler::Complex(a), Scaler::Float(b)) => a == b,
            (Scaler::Complex(a), Scaler::Complex(b)) => a == b,
        }
    }
}

impl PartialOrd for Scaler {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => a.partial_cmp(b),
            (Scaler::Int(a), Scaler::Float(b)) => a.partial_cmp(b),
            (Scaler::Int(a), Scaler::Complex(b)) => (a == b).then_some(Ordering::Equal),

            (Scaler::Float(a), Scaler::Int(b)) => a.partial_cmp(b),
            (Scaler::Float(a), Scaler::Float(b)) => a.partial_cmp(b),
            (Scaler::Float(a), Scaler::Complex(b)) => (a == b).then_some(Ordering::Equal),

            (Scaler::Complex(a), Scaler::Int(b)) => (a == b).then_some(Ordering::Equal),
            (Scaler::Complex(a), Scaler::Float(b)) => (a == b).then_some(Ordering::Equal),
            (Scaler::Complex(a), Scaler::Complex(b)) => (a == b).then_some(Ordering::Equal),
        }
    }
}

impl Pow<Scaler> for Scaler {
    type Output = Self;

    fn pow(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => {
                let fa = a.clone() * Complex::with_val(FLOAT_PRECISION, (1, 0));
                if b.is_integer() {
                    if let Some(b) = b.numer().to_u32() {
                        Scaler::from(a.pow(b))
                    } else {
                        let fb = b * Complex::with_val(FLOAT_PRECISION, (1, 0));
                        Scaler::from(fa.pow(fb))
                    }
                } else if b == (1, 2) {
                    Scaler::from(fa.sqrt())
                } else {
                    let fb = b * Complex::with_val(FLOAT_PRECISION, (1, 0));
                    Scaler::from(fa.pow(fb))
                }
            }
            (Scaler::Int(a), Scaler::Float(b)) => {
                let fa = a * Complex::with_val(FLOAT_PRECISION, (1, 0));
                let fb = Complex::from(b);
                Scaler::from(fa.pow(fb))
            }
            (Scaler::Int(a), Scaler::Complex(b)) => {
                let f = a * Complex::with_val(FLOAT_PRECISION, (1, 0));
                Scaler::from(f.pow(b))
            }

            (Scaler::Float(a), Scaler::Int(b)) => {
                let f = b * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(Complex::from(a).pow(f))
            }
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(Complex::from(a).pow(b)),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(Complex::from(a).pow(b)),

            (Scaler::Complex(a), Scaler::Int(b)) => {
                let f: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(a.pow(f))
            }
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a.pow(b)),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a.pow(b)),
        }
    }
}

impl ops::Rem for Scaler {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        if other.is_zero() {
            // @@ Infinity vs NegInfinity
            Scaler::Float(Float::with_val(1, Special::Infinity))
        } else {
            match (self, other) {
                (Scaler::Int(a), Scaler::Int(b)) => {
                    let af: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                    let bf: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                    Scaler::from(af % bf)
                }
                (Scaler::Int(a), Scaler::Float(b)) => {
                    let f: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                    Scaler::from(f % b)
                }
                (Scaler::Int(a), Scaler::Complex(b)) => {
                    let af: Float = a.clone() * Float::with_val(FLOAT_PRECISION, 1.0);
                    let mut c = b.clone();
                    c.div_from_round(af, (Round::Zero, Round::Zero));
                    Scaler::from(a - c * b)
                }

                (Scaler::Float(a), Scaler::Int(b)) => {
                    let bf: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                    Scaler::from(a % bf)
                }
                (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a % b),
                (Scaler::Float(a), Scaler::Complex(b)) => {
                    let mut c = b.clone();
                    c.div_from_round(a.clone(), (Round::Zero, Round::Zero));
                    Scaler::from(a - c * b)
                }

                (Scaler::Complex(a), Scaler::Int(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scaler::from(a - c * b)
                }
                (Scaler::Complex(a), Scaler::Float(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scaler::from(a - c * b)
                }
                (Scaler::Complex(a), Scaler::Complex(b)) => {
                    let mut c = a.clone();
                    c.div_assign_round(b.clone(), (Round::Zero, Round::Zero));
                    Scaler::from(a - c * b)
                }
            }
        }
    }
}

impl Signed for Scaler {
    fn abs(&self) -> Self {
        match self {
            Scaler::Int(x) => x.clone().abs().into(),
            Scaler::Float(x) => x.clone().abs().into(),
            Scaler::Complex(x) => x.clone().abs().into(),
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
            Scaler::Int(x) => x.clone().signum().into(),
            Scaler::Float(x) => x.clone().signum().into(),
            Scaler::Complex(x) => Scaler::from(Complex::with_val(
                1,
                (x.real().clone().signum(), x.imag().clone().signum()),
            )),
        }
    }
    fn is_positive(&self) -> bool {
        match self {
            Scaler::Int(x) => x.clone().signum() == 1,
            Scaler::Float(x) => x.is_sign_positive(),
            Scaler::Complex(x) => x.real().is_sign_positive(),
        }
    }
    fn is_negative(&self) -> bool {
        match self {
            Scaler::Int(x) => x.clone().signum() == -1,
            Scaler::Float(x) => x.is_sign_negative(),
            Scaler::Complex(x) => x.real().is_sign_negative(),
        }
    }
}

impl ops::Sub<Scaler> for Scaler {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a - b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a - b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a - b),
        }
    }
}

impl ops::SubAssign for Scaler {
    fn sub_assign(&mut self, other: Self) {
        *self = match (self.clone(), other) {
            (Scaler::Int(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Int(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Int(a), Scaler::Complex(b)) => Scaler::from(a - b),

            (Scaler::Float(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Float(a), Scaler::Complex(b)) => Scaler::from(a - b),

            (Scaler::Complex(a), Scaler::Int(b)) => Scaler::from(a - b),
            (Scaler::Complex(a), Scaler::Float(b)) => Scaler::from(a - b),
            (Scaler::Complex(a), Scaler::Complex(b)) => Scaler::from(a - b),
        }
    }
}

impl std::iter::Sum for Scaler {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Scaler>,
    {
        let mut a = Scaler::zero();

        for (_, b) in iter.enumerate() {
            a += b;
        }
        a
    }
}

impl Default for Scaler {
    fn default() -> Self {
        Scaler::zero()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn scaler_one_equality() {
        let a = Scaler::one();
        let b = Scaler::Float(Float::with_val(FLOAT_PRECISION, 1.0));
        let c = Scaler::Complex(Complex::with_val(FLOAT_PRECISION, (1.0, 0)));

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
        assert!(b.is_one());
        assert!(c.is_one());
    }

    #[test]
    fn scaler_zero_equality() {
        let a = Scaler::zero();
        let b = Scaler::Float(Float::with_val(FLOAT_PRECISION, 0.0));
        let c = Scaler::Complex(Complex::with_val(FLOAT_PRECISION, (0.0, 0.0)));

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
        assert!(b.is_zero());
        assert!(c.is_zero());
    }

    #[test]
    fn scaler_factor() {
        let a = Scaler::from(12);
        let f2 = Scaler::from(2);
        let f3 = Scaler::from(3);

        assert_eq!(a.factor(), Ok(vec![f2.clone(), f2, f3]));
    }

    #[test]
    fn value_get_usize_some() {
        let a = Value::from(Scaler::from(rug::Rational::from((5, 1))));
        assert_eq!(a.get_usize(), Some(5));
    }

    #[test]
    fn value_get_usize_none() {
        let b = Value::from(Scaler::from(rug::Rational::from((5, 2))));
        assert_eq!(b.get_usize(), None);
    }

    #[test]
    fn scaler_from_str_complex() {
        let a = Scaler::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, 2.0)));
        let b = Scaler::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -2.0)));

        assert_eq!(Scaler::try_from("(0 2)"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0+2i"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0 +2i"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0 + 2i"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0b0.0 + 0x2i"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("2i"), Ok(a));
        assert_eq!(Scaler::try_from("(0 -2)"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("0-2i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("0 -2i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("0 - 2i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("-0-2i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("+0-2i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("+0o0-0d2.0i"), Ok(b.clone()));
        assert_eq!(Scaler::try_from("-2i"), Ok(b));
    }

    #[test]
    fn scaler_from_str_rational() {
        let a = Scaler::from(rug::Rational::from((1, 2)));

        assert_eq!(Scaler::try_from("0"), Ok(Scaler::from(0)));
        assert_eq!(Scaler::try_from("-1"), Ok(Scaler::from(-1)));
        assert_eq!(Scaler::try_from("+1"), Ok(Scaler::from(1)));
        assert_eq!(Scaler::try_from("1/2"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0x1/0x2"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0b1/2"), Ok(a.clone()));
        assert_eq!(Scaler::try_from("0b1/0o2"), Ok(a));
    }

    #[test]
    fn scaler_to_string_radix() {
        let a = Scaler::from(rug::Complex::with_val(FLOAT_PRECISION, (10.0, 20.0)));
        let b = Scaler::from(rug::Complex::with_val(FLOAT_PRECISION, (0.0, -21.0)));
        let c = Scaler::try_from("1/2").unwrap();

        assert_eq!(a.to_string_radix(Radix::Decimal, true).as_str(), "10+20i");
        assert_eq!(b.to_string_radix(Radix::Decimal, true).as_str(), "-21i");
        assert_eq!(c.to_string_radix(Radix::Decimal, true).as_str(), "1/2");
        assert_eq!(a.to_string_radix(Radix::Decimal, false).as_str(), "10+20i");
        assert_eq!(b.to_string_radix(Radix::Decimal, false).as_str(), "-21i");
        assert_eq!(c.to_string_radix(Radix::Decimal, false).as_str(), "0.5");
        assert_eq!(a.to_string_radix(Radix::Hex, true).as_str(), "0xa+0x14i");
        assert_eq!(b.to_string_radix(Radix::Hex, true).as_str(), "-0x15i");
        assert_eq!(c.to_string_radix(Radix::Hex, true).as_str(), "0x1/0x2");
        assert_eq!(c.to_string_radix(Radix::Hex, false).as_str(), "0x0.8");
    }

    #[test]
    fn string_scaler_float() {
        let a = rug::Float::with_val(FLOAT_PRECISION, 1.5);
        let b = rug::Float::with_val(FLOAT_PRECISION, -1.5);
        let c = rug::Float::with_val(FLOAT_PRECISION, 0.25);
        let d = rug::Float::with_val(FLOAT_PRECISION, 16.0);
        assert_eq!(a.to_string_scaler(Radix::Decimal), "1.5");
        assert_eq!(a.to_string_scaler(Radix::Hex), "0x1.8");
        assert_eq!(a.to_string_scaler(Radix::Octal), "0o1.4");
        assert_eq!(a.to_string_scaler(Radix::Binary), "0b1.1");
        assert_eq!(b.to_string_scaler(Radix::Decimal), "-1.5");
        assert_eq!(b.to_string_scaler(Radix::Hex), "-0x1.8");
        assert_eq!(b.to_string_scaler(Radix::Octal), "-0o1.4");
        assert_eq!(b.to_string_scaler(Radix::Binary), "-0b1.1");
        assert_eq!(c.to_string_scaler(Radix::Decimal), "0.25");
        assert_eq!(c.to_string_scaler(Radix::Hex), "0x0.4");
        assert_eq!(c.to_string_scaler(Radix::Octal), "0o0.2");
        assert_eq!(c.to_string_scaler(Radix::Binary), "0b0.01");
        assert_eq!(d.to_string_scaler(Radix::Decimal), "16");
        assert_eq!(d.to_string_scaler(Radix::Hex), "0x10");
        assert_eq!(d.to_string_scaler(Radix::Octal), "0o20");
        assert_eq!(d.to_string_scaler(Radix::Binary), "0b10000");
    }

    #[test]
    fn string_scaler_rational() {
        let a = rug::Rational::from((-10, 1));
        let b = rug::Rational::from((10, 11));
        assert_eq!(a.to_string_scaler(Radix::Decimal), "-10");
        assert_eq!(a.to_string_scaler(Radix::Hex), "-0xa");
        assert_eq!(a.to_string_scaler(Radix::Octal), "-0o12");
        assert_eq!(a.to_string_scaler(Radix::Binary), "-0b1010");
        assert_eq!(b.to_string_scaler(Radix::Decimal), "10/11");
        assert_eq!(b.to_string_scaler(Radix::Hex), "0xa/0xb");
        assert_eq!(b.to_string_scaler(Radix::Octal), "0o12/0o13");
        assert_eq!(b.to_string_scaler(Radix::Binary), "0b1010/0b1011");
    }
}
