/* RPN-rs (c) 2026 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::numbers::Value;

#[derive(Debug, PartialEq)]
pub(crate) enum Return {
    Ok,
    Noop,
    Err(String),
}

pub(crate) trait StackOps {
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

#[cfg(test)]
mod tests {
    use super::{Return, StackOps as _};
    use crate::numbers::Value;
    use test_case::test_case;

    #[test_case(vec![Value::from(1)], Return::Ok; "Basic Data")]
    #[test_case(vec![], Return::Noop; "No Data")]
    fn unary(mut vv: Vec<Value>, rt: Return) {
        let vv2 = vv.clone();

        assert_eq!(vv.unary(|v| v), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1)], Return::Ok; "Basic Data")]
    #[test_case(vec![], Return::Noop; "No Data")]
    fn unary_v(mut vv: Vec<Value>, rt: Return) {
        let vv2 = vv.clone();

        assert_eq!(vv.unary_v(|v| vec![v]), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1)], Return::Ok; "Basic Data")]
    #[test_case(vec![], Return::Noop; "No Data")]
    fn try_unary_okay(mut vv: Vec<Value>, rt: Return) {
        let vv2 = vv.clone();

        assert_eq!(vv.try_unary(Ok), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1)], Return::Err("Bad Data".to_string()), "Bad Data"; "Basic Data")]
    #[test_case(vec![], Return::Noop, "Bad Data"; "No Data")]
    fn try_unary_err(mut vv: Vec<Value>, rt: Return, err: &str) {
        let vv2 = vec![];

        assert_eq!(vv.try_unary(|_v| Err(err.to_string())), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1)], Return::Ok; "Basic Data")]
    #[test_case(vec![], Return::Noop; "No Data")]
    fn try_unary_v_okay(mut vv: Vec<Value>, rt: Return) {
        let vv2 = vv.clone();

        assert_eq!(vv.try_unary_v(|a| Ok(vec![a])), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1)], Return::Err("Bad Data".to_string()), "Bad Data"; "Basic Data")]
    #[test_case(vec![], Return::Noop, "Bad Data"; "No Data")]
    fn try_unary_v_err(mut vv: Vec<Value>, rt: Return, err: &str) {
        let vv2 = vec![];

        assert_eq!(vv.try_unary_v(|_v| Err(err.to_string())), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], Return::Ok; "Basic Data")]
    #[test_case(vec![Value::from(1)], Return::Noop; "Not Enough Data")]
    #[test_case(vec![], Return::Noop; "No Data")]
    fn binary_v_data(mut vv: Vec<Value>, rt: Return) {
        let vv2 = vv.clone();

        assert_eq!(vv.binary_v(|a, b| vec![a, b]), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![Value::from(3)], Return::Ok; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![Value::from(1)], Return::Noop; "Not Enough Data")]
    #[test_case(vec![], vec![], Return::Noop; "No Data")]
    fn try_binary_okay(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return) {
        assert_eq!(vv.try_binary(|a, b| a + b), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![], Return::Err("Bad Data".to_string()), "Bad Data"; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![Value::from(1)], Return::Noop, "Bad Result"; "Not Enough Data")]
    #[test_case(vec![], vec![], Return::Noop, "Bad Data"; "No Data")]
    fn try_binary_err(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return, err: &str) {
        assert_eq!(vv.try_binary(|_a, _b| Err(err.to_string())), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![Value::from(3)], Return::Ok; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![Value::from(1)], Return::Ok; "Single Data")]
    #[test_case(vec![], vec![], Return::Noop; "No Data")]
    fn try_reduce_okay(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return) {
        assert_eq!(vv.try_reduce(|a, b| a + b), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![], Return::Err("Bad Data".to_string()), "Bad Data"; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![Value::from(1)], Return::Ok, "Bad Result"; "Single Data")]
    #[test_case(vec![], vec![], Return::Noop, "Bad Data"; "No Data")]
    fn try_reduce_err(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return, err: &str) {
        assert_eq!(vv.try_reduce(|_a, _b| Err(err.to_string())), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![Value::from(3)], Return::Ok; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![Value::from(1)], Return::Ok; "Single Data")]
    #[test_case(vec![], vec![Value::from(0)], Return::Ok; "No Data")]
    fn try_fold_okay(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return) {
        assert_eq!(vv.try_fold(Value::from(0), |a, b| a + b), rt);
        assert_eq!(vv, vv2);
    }

    #[test_case(vec![Value::from(1), Value::from(2)], vec![Value::from(1)], Return::Err("Bad Data".to_string()), "Bad Data"; "Basic Data")]
    #[test_case(vec![Value::from(1)], vec![], Return::Err("Bad Data".to_string()), "Bad Data"; "Single Data")]
    #[test_case(vec![], vec![Value::from(0)], Return::Ok, "Bad Data"; "No Data")]
    fn try_fold_err(mut vv: Vec<Value>, vv2: Vec<Value>, rt: Return, err: &str) {
        assert_eq!(
            vv.try_fold(Value::from(0), |_a, _b| Err(err.to_string())),
            rt
        );
        assert_eq!(vv, vv2);
    }
}
