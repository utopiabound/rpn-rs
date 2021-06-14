use rug::{float::Special, ops::RemRounding, Complex, Float, Integer, Rational};
use std::{
    convert::{From, TryFrom},
    ops,
};

const FLOAT_PRESITION: u32 = 53;

#[derive(Debug, Clone, PartialEq)]
pub enum ScalerValue {
    Int(Rational),
    Float(Float),
    Complex(Complex),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub values: Vec<Vec<ScalerValue>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(Rational),
    Float(Float),
    Complex(Complex),
    Matrix(Matrix),
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

impl From<Integer> for Value {
    fn from(x: Integer) -> Self {
        Value::Int(Rational::from(x))
    }
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
    pub fn is_zero(&self) -> bool {
        match self {
            Value::Int(x) => (x.numer().to_u32() == Some(0)),
            Value::Float(x) => x.is_zero(),
            Value::Complex(x) => x.real().is_zero(),
            Value::Matrix(_) => false, // @@
        }
    }

    pub fn to_string_radix(&self, radix: Radix, rational: bool) -> String {
        match self {
            Value::Int(x) => {
                if !rational && x.denom().to_u32() != Some(1) {
                    if radix == Radix::Decimal {
                        format!("{}", x.to_f64())
                    } else {
                        // @@
                        let f: Float = x * Float::with_val(53, 1.0);
                        f.to_string_radix(radix.into(), None)
                    }
                } else {
                    x.to_string_radix(radix.into())
                }
            }
            Value::Float(x) => {
                if x.is_normal() && radix == Radix::Decimal {
                    format!("{}", x.to_f64())
                } else {
                    x.to_string_radix(radix.into(), None)
                }
            }
            Value::Complex(x) => {
                let r = x.real();
                let i = x.imag();
                // @@ RADIX && check Float is normal
                format!("({}, {})", r.to_f64(), i.to_f64())
            }
            Value::Matrix(_x) => "NYI".to_string(),
        }
    }
    pub fn try_modulo(&self, b: &Value) -> Result<Value, String> {
        if b.is_zero() {
            return Err(format!("Division by zero"));
        }
        if let (Value::Int(a), Value::Int(b)) = (self, b) {
            if a.denom().to_u32() == Some(1) && b.denom().to_u32() == Some(1) {
                return Ok(Value::from(a.numer().clone().rem_euc(b.numer())));
            }
        }
        Err(format!("{:?} mod {:?} is not INT mod INT", self, b))
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
        if other.is_zero() {
            return Value::Float(Float::with_val(1, Special::Nan));
        }
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
