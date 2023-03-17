/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::numbers::{Radix, Value};
use std::error::Error;

// possibly optional
pub mod fltk;
pub mod readline;

#[derive(clap::ValueEnum, Default, Debug, Copy, Clone, strum::Display, PartialEq)]
#[clap(rename_all = "lower")]
#[strum(serialize_all = "lowercase")]
pub enum Flavor {
    /// Graphical User Interface
    #[default]
    Gui,
    /// Text User Interface
    Cli,
}

pub fn get_ui(flavor: Flavor) -> Result<Box<dyn CalcDisplay>, Box<dyn Error + Send + Sync>> {
    match flavor {
        Flavor::Gui => Ok(Box::new(fltk::FltkCalcDisplay::init()?)),
        Flavor::Cli => Ok(Box::new(readline::ReadlineCalcUI::init()?)),
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    Clear,
    Drop,
    Input(String),
}

pub trait CalcDisplay {
    /// Initialize Display driver
    fn init() -> Result<Self, Box<dyn Error + Send + Sync>>
    where
        Self: Sized;

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
