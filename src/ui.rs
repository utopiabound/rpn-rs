/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::numbers::{Radix, Value};

// possibly optional
pub mod fltk;
// pub mod cli;

#[derive(Debug, Clone)]
pub enum Message {
    Clear,
    Drop,
    Input(String),
}

pub trait CalcDisplay {
    /// Initialize Display driver
    fn init() -> Self where Self: Sized;

    /// Wait on next Message for event loop
    fn next(&mut self) -> Option<Message>;

    /// Push input around again
    fn push_input(&mut self, value: String);

    /// Set/Clear error message
    fn set_error(&mut self, msg: Option<String>);

    /// Display Dialog with text
    fn dialog(&self, mgs: String);

    fn set_data(&mut self, newdata: &[Value]);

    fn set_display(&mut self, radix: Option<Radix>, rational: Option<bool>);

    /// Show Help Text
    fn help(&mut self);

    /// Cleanup and quit
    fn quit(&self);
}
