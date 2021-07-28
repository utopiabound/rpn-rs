/* RPN-rs (c) 2021 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use regex::Regex;
use rug::{
    float::{Constant, Special},
    ops::{Pow, RemRounding},
    Complex, Float, Integer, Rational,
};
use std::{
    convert::{From, TryFrom},
    ops,
};

const FLOAT_PRECISION: u32 = 256;

#[derive(Debug, Clone, PartialEq)]
pub enum Scaler {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub values: Vec<Vec<Scaler>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scaler(Scaler),
    Matrix(Matrix),
}

impl From<Matrix> for Value {
    fn from(x: Matrix) -> Self {
        Value::Matrix(x)
    }
}

impl<T: Into<Scaler>> From<T> for Value {
    fn from(x: T) -> Self {
        Value::Scaler(x.into())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Radix {
    Decimal,
    Hex,
    Binary,
    Octal,
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

impl TryFrom<&str> for Value {
    type Error = String;

    // @@ Add radix parsing
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let radixre = Regex::new(r"0([xXoObB])(.*)").unwrap();

        if value.contains(&['(', 'i'][..]) {
            // @@ better float parsing (pi, e)

            // Change "1+2i" -> "(1 2)"
            let re = Regex::new(r"(?P<r>\d+([.]\d+)?)\s*\+\s*(?P<i>\d+([.]\d+)?)i").unwrap();
            let value = re.replace(value, "($r $i)").into_owned();
            // Change "3i" -> "(0 3)"
            let re = Regex::new(r"(?P<i>\d+([.]\d+)?)i").unwrap();
            let value = re.replace(&value, "(0 $i)").into_owned();

            let v = Complex::parse(&value).map_err(|e| e.to_string())?;
            let c = Complex::with_val(FLOAT_PRECISION, v);
            Ok(Value::from(c))
        } else if value.contains("[") {
            Err(format!("Matrix NYI"))
        } else if value.contains(".") {
            let v = Float::parse(&value).map_err(|e| e.to_string())?;
            let f = Float::with_val(FLOAT_PRECISION, v);
            Ok(Value::from(f))
        } else if let Some(caps) = radixre.captures(value) {
            match caps[1].to_lowercase().as_str() {
                "b" => Rational::parse_radix(caps[2].to_string(), 2)
                    .map(|v| Value::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "o" => Rational::parse_radix(caps[2].to_string(), 8)
                    .map(|v| Value::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "d" => Rational::parse_radix(caps[2].to_string(), 10)
                    .map(|v| Value::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "x" => Rational::parse_radix(caps[2].to_string(), 16)
                    .map(|v| Value::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                r => Err(format!("Invalid radix {} in {}", r, &value)),
            }
        } else if let Ok(v) = Rational::parse(&value) {
            Ok(Value::from(Rational::from(v)))
        } else {
            Err(format!("Unknown value: {}", &value))
        }
    }
}

impl Scaler {
    /// True if value is Zero
    pub fn is_zero(&self) -> bool {
        match self {
            Scaler::Int(x) => x.numer().to_u32() == Some(0),
            Scaler::Float(x) => x.is_zero(),
            Scaler::Complex(x) => x.real().is_zero(),
        }
    }

    /// Return Square Root of value
    pub fn sqrt(self) -> Self {
        match self {
            Scaler::Int(x) => {
                // @@ - try to keep as rational
                let f: Float = x * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(Complex::from(f).sqrt())
            }
            Scaler::Float(x) => Scaler::from(Complex::from(x).sqrt()),
            Scaler::Complex(x) => Scaler::from(x.sqrt()),
        }
    }

    /// Retrurn Natural Logorithm of value
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

    /// Retrurn Logorithm of value
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

    /// Return Inverse of value "1/x"
    pub fn inverse(&self) -> Self {
        match self {
            Scaler::Int(x) => Scaler::from(Rational::from((x.denom(), x.numer()))),
            Scaler::Float(x) => Scaler::from(Float::with_val(FLOAT_PRECISION, (1.0 as f32) / x)),
            Scaler::Complex(x) => {
                Scaler::from(Complex::with_val(FLOAT_PRECISION, (1.0 as f32) / x))
            }
        }
    }
}

impl Value {
    pub fn is_zero(&self) -> bool {
        match self {
            Value::Scaler(x) => x.is_zero(),
            Value::Matrix(_) => false, // @@
        }
    }

    pub fn sqrt(self) -> Self {
        match self {
            Value::Scaler(x) => Value::from(x.sqrt()),
            Value::Matrix(_) => {
                // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
                // https://en.wikipedia.org/wiki/Square_root_of_a_matrix
                Value::Scaler(Scaler::Float(Float::with_val(1, Special::Nan)))
            }
        }
    }

    pub fn to_string_radix(&self, radix: Radix, rational: bool) -> String {
        match self {
            Value::Scaler(Scaler::Int(x)) => {
                if !rational && x.denom().to_u32() != Some(1) {
                    if radix == Radix::Decimal {
                        format!("{}", x.to_f64())
                    } else {
                        // @@
                        let f: Float = x * Float::with_val(FLOAT_PRECISION, 1.0);
                        f.to_string_radix(radix.into(), None)
                    }
                } else {
                    x.to_string_radix(radix.into())
                }
            }
            Value::Scaler(Scaler::Float(x)) => {
                if x.is_normal() && radix == Radix::Decimal {
                    format!("{}", x.to_f64())
                } else {
                    x.to_string_radix(radix.into(), None)
                }
            }
            Value::Scaler(Scaler::Complex(x)) => {
                let r = x.real();
                let i = x.imag();
                // @@ RADIX && check Float is normal
                if r.is_zero() {
                    format!("{}i", i.to_f64())
                } else {
                    format!("{} + {}i", r.to_f64(), i.to_f64())
                }
            }
            Value::Matrix(_x) => "NYI".to_string(),
        }
    }

    pub fn try_modulo(&self, b: &Value) -> Result<Value, String> {
        if b.is_zero() {
            return Err(format!("Division by zero"));
        }
        if let (Value::Scaler(Scaler::Int(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if a.denom().to_u32() == Some(1) && b.denom().to_u32() == Some(1) {
                return Ok(Value::from(a.numer().clone().rem_euc(b.numer())));
            }
        }
        Err(format!("{:?} mod {:?} is not INT mod INT", self, b))
    }

    pub fn try_pow(self, other: Value) -> Result<Self, String> {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::from(a.pow(b))),
            _ => Err("NYI".to_string()),
        }
    }

    pub fn try_root(self, other: Value) -> Result<Self, String> {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::from(a.pow(b.inverse()))),
            _ => Err("NYI".to_string()),
        }
    }

    pub fn try_rshift(&self, b: &Value) -> Result<Self, String> {
        if let (Value::Scaler(Scaler::Int(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() >> b));
                }
            }
        } else if let (Value::Scaler(Scaler::Float(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() >> b));
                }
            }
        }
        Err(format!("{:?} >> {:?} is not REAL >> INTEGER(u32)", self, b))
    }

    pub fn try_lshift(&self, b: &Value) -> Result<Self, String> {
        if let (Value::Scaler(Scaler::Int(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::Scaler(Scaler::from(a.clone() << b)));
                }
            }
        } else if let (Value::Scaler(Scaler::Float(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() << b));
                }
            }
        }
        Err(format!("{:?} << {:?} is not REAL >> INTEGER(u32)", self, b))
    }

    pub fn try_ln(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.ln())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    pub fn try_log10(self) -> Result<Self, String> {
        match self {
            Value::Scaler(x) => Ok(Value::Scaler(x.log10())),
            Value::Matrix(_) => Err("NYI".to_string()),
        }
    }

    // Constants
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
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Value::Scaler(Scaler::from(a + b)),
            _ => Value::Scaler(Scaler::Float(Float::with_val(1, Special::Nan))),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Value::Scaler(Scaler::from(a - b)),
            _ => Value::Scaler(Scaler::Float(Float::with_val(1, Special::Nan))),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Value::Scaler(Scaler::from(a * b)),
            _ => Value::Scaler(Scaler::Float(Float::with_val(1, Special::Nan))),
        }
    }
}

impl ops::Div<Value> for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Value::Scaler(Scaler::from(a / b)),
            _ => Value::Scaler(Scaler::Float(Float::with_val(1, Special::Nan))),
        }
    }
}

impl ops::Add<Scaler> for Scaler {
    type Output = Self;

    fn add(self, other: Self) -> Self {
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

impl ops::Sub<Scaler> for Scaler {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
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

impl ops::Mul<Scaler> for Scaler {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
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

impl ops::Div<Scaler> for Scaler {
    type Output = Self;

    fn div(self, other: Self) -> Self {
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

impl Pow<Scaler> for Scaler {
    type Output = Self;

    fn pow(self, other: Self) -> Self {
        match (self, other) {
            (Scaler::Int(a), Scaler::Int(b)) => {
                // @@ attempt to maintain int
                let fa: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                let fb: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(fa.pow(fb))
            }
            (Scaler::Int(a), Scaler::Float(b)) => {
                let fa: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                let fb: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(fa.pow(fb))
            }
            (Scaler::Int(a), Scaler::Complex(b)) => {
                let f: Float = a * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(Complex::from(f).pow(b))
            }

            (Scaler::Float(a), Scaler::Int(b)) => {
                let f: Float = b * Float::with_val(FLOAT_PRECISION, 1.0);
                Scaler::from(a.pow(f))
            }
            (Scaler::Float(a), Scaler::Float(b)) => Scaler::from(a.pow(b)),
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
