use rug::{float::Special, Complex, Float, Integer, Rational};
use std::{fmt, ops};

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

impl Value {
    pub fn parse(value: &str) -> Result<Self, String> {
        if value.contains("(") {
            let v = Complex::parse(value).map_err(|e| e.to_string())?;
            Ok(Value::Complex(Complex::with_val(53, v)))
        } else if value.contains(".") {
            let v = Float::parse(value).map_err(|e| e.to_string())?;
            let f = Float::with_val(53, v);
            Ok(Value::Float(f))
        } else if let Ok(v) = Rational::parse(value) {
            Ok(Value::Int(Rational::from(v)))
        } else if let Ok(v) = Float::parse(value) {
            Ok(Value::Float(Float::with_val(53, v)))
        } else {
            Err(format!("Unknown value: {}", value))
        }
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
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(b + a),
            (Value::Int(a), Value::Complex(b)) => Value::Complex(a + b),

            (Value::Float(a), Value::Int(b)) => Value::Float(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Float(a), Value::Complex(b)) => Value::Complex(a + b),

            (Value::Complex(a), Value::Int(b)) => Value::Complex(a + b),
            (Value::Complex(a), Value::Float(b)) => Value::Complex(a + b),
            (Value::Complex(a), Value::Complex(b)) => Value::Complex(a + b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Complex(b)) => Value::Complex(a - b),

            (Value::Float(a), Value::Int(b)) => Value::Float(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Float(a), Value::Complex(b)) => Value::Complex(a - b),

            (Value::Complex(a), Value::Int(b)) => Value::Complex(a - b),
            (Value::Complex(a), Value::Float(b)) => Value::Complex(a - b),
            (Value::Complex(a), Value::Complex(b)) => Value::Complex(a - b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(b * a),
            (Value::Int(a), Value::Complex(b)) => Value::Complex(a * b),

            (Value::Float(a), Value::Int(b)) => Value::Float(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Float(a), Value::Complex(b)) => Value::Complex(a * b),

            (Value::Complex(a), Value::Int(b)) => Value::Complex(a * b),
            (Value::Complex(a), Value::Float(b)) => Value::Complex(a * b),
            (Value::Complex(a), Value::Complex(b)) => Value::Complex(a * b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}

impl ops::Div<Value> for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::Int(a), Value::Float(b)) => Value::Float(a / b),
            (Value::Int(a), Value::Complex(b)) => {
                let f: Float = a * Float::with_val(53, 1.0);
                Value::Complex(f / b)
            }

            (Value::Float(a), Value::Int(b)) => Value::Float(a / b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (Value::Float(a), Value::Complex(b)) => Value::Complex(a / b),

            (Value::Complex(a), Value::Int(b)) => Value::Complex(a / b),
            (Value::Complex(a), Value::Float(b)) => Value::Complex(a / b),
            (Value::Complex(a), Value::Complex(b)) => Value::Complex(a / b),

            _ => Value::Float(Float::with_val(1, Special::Nan)),
        }
    }
}
