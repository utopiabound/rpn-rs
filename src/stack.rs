/* RPN-rs (c) 2021 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::numbers::Value;

pub enum Return {
    Ok,
    Noop,
    Err(String),
}

pub trait StackOps {
    fn unary<F: Fn(Value) -> Value>(&mut self, f: F) -> Return;
    fn binary<F: Fn(Value, Value) -> Value>(&mut self, f: F) -> Return;
    fn try_binary<F: Fn(Value, Value) -> Result<Value, String>>(&mut self, f: F) -> Return;
    fn unary2<F: Fn(Value) -> (Value, Value)>(&mut self, f: F) -> Return;
    fn binary2<F: Fn(Value, Value) -> (Value, Value)>(&mut self, f: F) -> Return;
}

impl StackOps for Vec<Value> {
    fn unary<F: Fn(Value) -> Value>(&mut self, f: F) -> Return {
        if let Some(a) = self.pop() {
            let c = f(a);
            self.push(c);
            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn binary<F: Fn(Value, Value) -> Value>(&mut self, f: F) -> Return {
        if self.len() > 1 {
            let a = self.pop().unwrap();
            let b = self.pop().unwrap();
            let c = f(b, a);
            self.push(c);
            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn binary2<F: Fn(Value, Value) -> (Value, Value)>(&mut self, f: F) -> Return {
        if self.len() > 1 {
            let a = self.pop().unwrap();
            let b = self.pop().unwrap();
            let (c, d) = f(b, a);
            self.push(c);
            self.push(d);
            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn unary2<F: Fn(Value) -> (Value, Value)>(&mut self, f: F) -> Return {
        if let Some(a) = self.pop() {
            let (c, d) = f(a);
            self.push(c);
            self.push(d);
            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn try_binary<F: Fn(Value, Value) -> Result<Value, String>>(&mut self, f: F) -> Return {
        if self.len() > 1 {
            let a = self.pop().unwrap();
            let b = self.pop().unwrap();
            match f(b, a) {
                Ok(c) => {
                    self.push(c);
                    Return::Ok
                }
                Err(e) => Return::Err(e),
            }
        } else {
            Return::Noop
        }
    }
}
