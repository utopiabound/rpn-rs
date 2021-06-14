use rug::{float::Special, Complex, Float, Rational};
use std::{
    convert::{From, TryFrom},
    fmt, ops,
};

const FLOAT_PRESITION: u32 = 53;

#[derive(Debug, Clone)]
pub enum ScalerValue {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub values: Vec<Vec<ScalerValue>>,
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(Rational),
    Float(Float),
    Complex(Complex),
    Matrix(Matrix),
}

impl From<Rational> for Value {
    fn from(x: Rational) -> Self {
        Value::Int(x)
    }
}

impl From<Float> for Value {
    fn from(x: Float) -> Self {
        if x.is_integer() {
            Value::Int(Rational::from(x.to_integer().unwrap()))
        } else {
            Value::Float(x)
        }
    }
}

impl From<Complex> for Value {
    fn from(x: Complex) -> Self {
        if x.imag().is_zero() {
            Value::from(x.real().clone())
        } else {
            Value::Complex(x)
        }
    }
}

impl TryFrom<&str> for Value {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.contains("(") {
            let v = Complex::parse(value).map_err(|e| e.to_string())?;
            let c = Complex::with_val(FLOAT_PRESITION, v);
            Ok(Value::from(c))
        } else if value.contains("[") {
            Err(format!("Matrix NYI"))
        } else if value.contains(".") {
            let v = Float::parse(value).map_err(|e| e.to_string())?;
            let f = Float::with_val(FLOAT_PRESITION, v);
            Ok(Value::from(f))
        } else if let Ok(v) = Rational::parse(value) {
            Ok(Value::Int(Rational::from(v)))
        } else {
            Err(format!("Unknown value: {}", value))
        }
    }
}

impl Value {
    pub fn modulo(&self, b: &Value) -> Option<Value> {
        None
    }
    pub fn try_rshift(&self, b: &Value) -> Result<Self, String> {
        if let (Value::Int(a), Value::Int(b)) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() >> b));
                }
            }
        } else if let (Value::Float(a), Value::Int(b)) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() >> b));
                }
            }
        }
        Err(format!("{:?} >> {:?} is not REAL >> INTEGER(u32)", self, b))
    }
    pub fn try_lshift(&self, b: &Value) -> Result<Self, String> {
        if let (Value::Int(a), Value::Int(b)) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() << b));
                }
            }
        } else if let (Value::Float(a), Value::Int(b)) = (self, b) {
            if b.denom().to_u32() == Some(1) {
                if let Some(b) = b.numer().to_u32() {
                    return Ok(Value::from(a.clone() << b));
                }
            }
        }
        Err(format!("{:?} << {:?} is not REAL >> INTEGER(u32)", self, b))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(x) => write!(f, "{}", x),
            Value::Float(x) => {
                if x.is_normal() {
                    write!(f, "{}", x.to_f64())
                } else {
                    write!(f, "{}", x)
                }
            }
            Value::Complex(x) => {
                let r = x.real();
                let i = x.imag();
                write!(f, "({}, {})", r.to_f64(), i.to_f64())
            }
            Value::Matrix(_x) => write!(f, "NYI"),
        }
    }
}

impl ops::Add<Value> for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::from(a + b),
            (Value::Int(a), Value::Float(b)) => Value::from(a + b),
            (Value::Int(a), Value::Complex(b)) => Value::from(a + b),

            (Value::Float(a), Value::Int(b)) => Value::from(a + b),
            (Value::Float(a), Value::Float(b)) => Value::from(a + b),
            (Value::Float(a), Value::Complex(b)) => Value::from(a + b),

            (Value::Complex(a), Value::Int(b)) => Value::from(a + b),
            (Value::Complex(a), Value::Float(b)) => Value::from(a + b),
            (Value::Complex(a), Value::Complex(b)) => Value::from(a + b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::from(a - b),
            (Value::Int(a), Value::Float(b)) => Value::from(a - b),
            (Value::Int(a), Value::Complex(b)) => Value::from(a - b),

            (Value::Float(a), Value::Int(b)) => Value::from(a - b),
            (Value::Float(a), Value::Float(b)) => Value::from(a - b),
            (Value::Float(a), Value::Complex(b)) => Value::from(a - b),

            (Value::Complex(a), Value::Int(b)) => Value::from(a - b),
            (Value::Complex(a), Value::Float(b)) => Value::from(a - b),
            (Value::Complex(a), Value::Complex(b)) => Value::from(a - b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::from(a * b),
            (Value::Int(a), Value::Float(b)) => Value::from(b * a),
            (Value::Int(a), Value::Complex(b)) => Value::from(a * b),

            (Value::Float(a), Value::Int(b)) => Value::from(a * b),
            (Value::Float(a), Value::Float(b)) => Value::from(a * b),
            (Value::Float(a), Value::Complex(b)) => Value::from(a * b),

            (Value::Complex(a), Value::Int(b)) => Value::from(a * b),
            (Value::Complex(a), Value::Float(b)) => Value::from(a * b),
            (Value::Complex(a), Value::Complex(b)) => Value::from(a * b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Div<Value> for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::from(a / b),
            (Value::Int(a), Value::Float(b)) => Value::from(a / b),
            (Value::Int(a), Value::Complex(b)) => {
                let f: Float = a * Float::with_val(53, 1.0);
                Value::from(f / b)
            }

            (Value::Float(a), Value::Int(b)) => Value::from(a / b),
            (Value::Float(a), Value::Float(b)) => Value::from(a / b),
            (Value::Float(a), Value::Complex(b)) => Value::from(a / b),

            (Value::Complex(a), Value::Int(b)) => Value::from(a / b),
            (Value::Complex(a), Value::Float(b)) => Value::from(a / b),
            (Value::Complex(a), Value::Complex(b)) => Value::from(a / b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}
