/* RPN-rs (c) 2024 Nathaniel Clark
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
    fn try_unary<F: Fn(Value) -> Result<Value, String>>(&mut self, f: F) -> Return;
    //fn binary<F: Fn(Value, Value) -> Value>(&mut self, f: F) -> Return;
    fn try_binary<F: Fn(Value, Value) -> Result<Value, String>>(&mut self, f: F) -> Return;
    fn try_unary_v<F: Fn(Value) -> Result<Vec<Value>, String>>(&mut self, f: F) -> Return;
    fn unary_v<F: Fn(Value) -> Vec<Value>>(&mut self, f: F) -> Return;
    fn binary_v<F: Fn(Value, Value) -> Vec<Value>>(&mut self, f: F) -> Return;
    fn try_reduce<F: Fn(Value, Value) -> Result<Value, String>>(&mut self, f: F) -> Return;
    fn try_fold<F: Fn(Value, Value) -> Result<Value, String>>(
        &mut self,
        init: Value,
        f: F,
    ) -> Return;
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
    // fn binary<F: Fn(Value, Value) -> Value>(&mut self, f: F) -> Return {
    //     if self.len() > 1 {
    //         let a = self.pop().unwrap();
    //         let b = self.pop().unwrap();
    //         let c = f(b, a);
    //         self.push(c);
    //         Return::Ok
    //     } else {
    //         Return::Noop
    //     }
    // }
    fn binary_v<F: Fn(Value, Value) -> Vec<Value>>(&mut self, f: F) -> Return {
        if self.len() > 1 {
            let a = self.pop().unwrap();
            let b = self.pop().unwrap();
            let res = f(b, a);
            self.extend(res);

            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn unary_v<F: Fn(Value) -> Vec<Value>>(&mut self, f: F) -> Return {
        if let Some(a) = self.pop() {
            let res = f(a);
            self.extend(res);
            Return::Ok
        } else {
            Return::Noop
        }
    }
    fn try_unary<F: Fn(Value) -> Result<Value, String>>(&mut self, f: F) -> Return {
        if let Some(a) = self.pop() {
            match f(a) {
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
    fn try_unary_v<F: Fn(Value) -> Result<Vec<Value>, String>>(&mut self, f: F) -> Return {
        if let Some(a) = self.pop() {
            match f(a) {
                Ok(c) => {
                    self.extend(c);
                    Return::Ok
                }
                Err(e) => Return::Err(e),
            }
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
    // This is roughly Iterator::try_reduce() but needs to consume self
    fn try_reduce<F: Fn(Value, Value) -> Result<Value, String>>(&mut self, f: F) -> Return {
        if let Some(acc) = self.pop() {
            self.try_fold(acc, f)
        } else {
            Return::Noop
        }
    }
    // This is roughly Iterator::try_reduce() but needs to consume self
    fn try_fold<F: Fn(Value, Value) -> Result<Value, String>>(
        &mut self,
        init: Value,
        f: F,
    ) -> Return {
        let mut acc = init;
        while let Some(e) = self.pop() {
            match f(acc, e) {
                Ok(c) => acc = c,
                Err(e) => return Return::Err(e),
            }
        }
        self.push(acc);
        Return::Ok
    }
}
