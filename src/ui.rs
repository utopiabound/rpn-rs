/* RPN-rs (c) 2025 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::numbers::{Angle, Radix, Value};
use std::error::Error;

pub(crate) mod fltk;
pub(crate) mod readline;
pub(crate) mod tui;

pub(crate) const HELP_HTML: &str = include_str!("fixtures/help.html");

pub(crate) fn about_txt() -> String {
    format!(
        "RPN Calculator {} (c) 2025 {}",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_AUTHORS")
    )
}

#[derive(clap::ValueEnum, Default, Debug, Copy, Clone, strum_macros::Display, PartialEq)]
#[clap(rename_all = "lower")]
#[strum(serialize_all = "lowercase")]
pub(crate) enum Flavor {
    /// Graphical User Interface
    #[default]
    Gui,
    /// Simple Text User Interface
    Cli,
    /// Fancy Text User Interface
    Tui,
}

pub(crate) fn get_ui(flavor: Flavor) -> Result<Box<dyn CalcDisplay>, Box<dyn Error + Send + Sync>> {
    match flavor {
        Flavor::Gui => Ok(Box::new(fltk::FltkCalcDisplay::init()?)),
        Flavor::Cli => Ok(Box::new(readline::ReadlineCalcUI::init()?)),
        Flavor::Tui => Ok(Box::new(tui::TuiCalcUI::init()?)),
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Message {
    Clear,
    Drop,
    Input(String),
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Info {
    pub(crate) radix: Radix,
    pub(crate) rational: bool,
    pub(crate) angle: Angle,
}

impl Default for Info {
    fn default() -> Self {
        Self {
            radix: Radix::default(),
            rational: true,
            angle: Angle::default(),
        }
    }
}

pub(crate) trait CalcDisplay {
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
    fn dialog(&mut self, mgs: String);

    fn set_data(&mut self, newdata: &[Value]);

    fn set_info(&mut self, info: Info);

    fn get_info(&self) -> Info;

    /// Show Help Text
    fn help(&mut self);

    /// Show About Text
    fn about(&mut self);

    /// Cleanup and quit
    fn quit(&mut self);
}

pub(crate) fn help_text(width: usize) -> String {
    html2text::from_read(HELP_HTML.as_bytes(), width)
        .unwrap_or_else(|e| format!("Failed to render help: {e}"))
}
