/* RPN-rs (c) 2021 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use libmat::mat::Matrix;
use num_traits::{Inv, Num, One, Signed, Zero};
use regex::Regex;
use rug::{
    float::{Constant, Round, Special},
    ops::{DivAssignRound, DivFromRound, Pow, RemRounding},
    Complex, Float, Integer, Rational,
};
use std::{cmp::Ordering, ops};

const FLOAT_PRECISION: u32 = 256;

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

impl TryFrom<&str> for Scaler {
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
            Ok(Scaler::from(c))
        } else if value.contains('[') {
            Err("Parsing Matrix as Scaler".to_string())
        } else if value.contains('.') {
            let v = Float::parse(&value).map_err(|e| e.to_string())?;
            let f = Float::with_val(FLOAT_PRECISION, v);
            Ok(Scaler::from(f))
        } else if let Some(caps) = radixre.captures(value) {
            match caps[1].to_lowercase().as_str() {
                "b" => Rational::parse_radix(caps[2].to_string(), 2)
                    .map(|v| Scaler::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "o" => Rational::parse_radix(caps[2].to_string(), 8)
                    .map(|v| Scaler::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "d" => Rational::parse_radix(caps[2].to_string(), 10)
                    .map(|v| Scaler::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                "x" => Rational::parse_radix(caps[2].to_string(), 16)
                    .map(|v| Scaler::from(Rational::from(v)))
                    .map_err(|e| e.to_string()),
                r => Err(format!("Invalid radix {} in {}", r, &value)),
            }
        } else if let Ok(v) = Rational::parse(&value) {
            Ok(Scaler::from(Rational::from(v)))
        } else {
            Err(format!("Unknown value: {}", &value))
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
    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("NYI".to_string()) // @@
    }
}

fn is_integer(x: &Rational) -> bool {
    x.denom().to_u32() == Some(1)
}

impl Scaler {
    /// Square Root
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
                } else if !is_integer(&x) {
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

    fn get_usize(self) -> Option<usize> {
        if let Scaler::Int(x) = self {
            if is_integer(&x) {
                x.numer().to_usize()
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn to_string_radix(&self, radix: Radix, rational: bool) -> String {
        match self {
            Scaler::Int(x) => {
                if !rational && !is_integer(x) {
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
            Scaler::Float(x) => {
                if x.is_normal() && radix == Radix::Decimal {
                    format!("{}", x.to_f64())
                } else {
                    x.to_string_radix(radix.into(), None)
                }
            }
            Scaler::Complex(x) => {
                let r = x.real();
                let i = x.imag();
                // @@ RADIX && check Float is normal
                if r.is_zero() {
                    format!("{}i", i.to_f64())
                } else {
                    format!("{}+{}i", r.to_f64(), i.to_f64())
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
    pub fn is_zero(&self) -> bool {
        match self {
            Value::Scaler(x) => x.is_zero(),
            Value::Matrix(x) => {
                for i in 0..x.row_count() {
                    for j in 0..x.col_count() {
                        if !x[i][j].is_zero() {
                            return false;
                        }
                    }
                }
                true
            }
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

    pub fn to_string_radix(&self, radix: Radix, rational: bool, flat: bool) -> String {
        match self {
            Value::Scaler(x) => x.to_string_radix(radix, rational),
            Value::Matrix(x) => {
                let mut s = String::from("[");
                let rowmax = x.row_count();
                for i in 0..rowmax {
                    for j in 0..x.col_count() {
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
            Value::Matrix(x) => x.row_count(),
        }
    }

    pub fn try_factor(self) -> Result<Vec<Value>, String> {
        match self {
            Value::Scaler(x) => x
                .factor()
                .map(|x| x.into_iter().map(Value::Scaler).collect()),
            Value::Matrix(_) => Err("No Factor for Matricies".to_string()), // @@ is this true?
        }
    }

    pub fn try_modulo(&self, b: &Value) -> Result<Value, String> {
        if b.is_zero() {
            return Err("Division by zero".to_string());
        }
        if let (Value::Scaler(Scaler::Int(a)), Value::Scaler(Scaler::Int(b))) = (self, b) {
            if a.denom().to_u32() == Some(1) && b.denom().to_u32() == Some(1) {
                return Ok(Value::from(a.numer().clone().rem_euc(b.numer())));
            }
        }
        Err(format!("{:?} mod {:?} is not INT mod INT", self, b))
    }

    pub fn try_root(self, other: Value) -> Result<Self, String> {
        match (self, other) {
            (Value::Scaler(a), Value::Scaler(b)) => Ok(Value::from(a.pow(b.inv()))),
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

    // Matrix Only Functions
    pub fn try_det(self) -> Result<Self, String> {
        match self {
            Value::Scaler(_) => Err("No determinate for Scaler".to_string()),
            Value::Matrix(x) => Ok(Value::Scaler(x.det().map_err(|e| e.to_string())?)),
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
                    Matrix::one(x.row_count())
                        .map(Value::Matrix)
                        .map_err(|e| e.to_string())
                } else {
                    Err("Matrix not square, cannot convert to identity".to_string())
                }
            }
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
            _ => Err("Illegal Operation: Scaler & Matrix Addition".to_string()),
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
            Value::Scaler(x) => Ok(Value::Scaler(x.inv())),
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
            (Value::Matrix(_m), Value::Scaler(b)) => match b {
                Scaler::Int(x) => {
                    let (mut num, den) = x.into_numer_denom();
                    if den != 1 {
                        Err(format!("NYI (denom = {})", den))
                    } else {
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
                    }
                }
                _ => Err("Unsupported: Matrix raised to non-whole power".to_string()),
            },
            (_, Value::Matrix(_)) => Err("Unsupported: matrix as exponent".to_string()),
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
            _ => Err("Illegal Operation: Matrix & Scaler Subtraction".to_string()),
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
            Scaler::Float(x) => Scaler::from(Float::with_val(FLOAT_PRECISION, (1.0_f32) / x)),
            Scaler::Complex(x) => Scaler::from(Complex::with_val(FLOAT_PRECISION, (1.0_f32) / x)),
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
            (Scaler::Int(a), Scaler::Complex(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }

            (Scaler::Float(a), Scaler::Int(b)) => a.partial_cmp(b),
            (Scaler::Float(a), Scaler::Float(b)) => a.partial_cmp(b),
            (Scaler::Float(a), Scaler::Complex(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }

            (Scaler::Complex(a), Scaler::Int(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (Scaler::Complex(a), Scaler::Float(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            (Scaler::Complex(a), Scaler::Complex(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
        }
    }
}

impl Pow<Scaler> for Scaler {
    type Output = Self;

    fn pow(self, other: Self) -> Self::Output {
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
        // @@ if self < other
        // return 0
        // else
        self.clone() - other.clone()
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
            Scaler::Complex(_) => false, // @@
        }
    }
    fn is_negative(&self) -> bool {
        match self {
            Scaler::Int(x) => x.clone().signum() == -1,
            Scaler::Float(x) => x.is_sign_negative(),
            Scaler::Complex(_) => false, // @@
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

#[cfg(test)]
mod test {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_scaler_one_equality() {
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
    fn test_scaler_zero_equality() {
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
    fn test_factor() {
        let a = Scaler::from(12);
        let f2 = Scaler::from(2);
        let f3 = Scaler::from(3);

        assert_eq!(a.factor(), Ok(vec![f2.clone(), f2, f3]));
    }
}
