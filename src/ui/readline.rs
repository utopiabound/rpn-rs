/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::{
    numbers::{Radix, Value},
    ui::{CalcDisplay, Message},
};

use rustyline::{error::ReadlineError, DefaultEditor};

pub struct ReadlineCalcUI {
    radix: Radix,
    rational: bool,
    editor: DefaultEditor,
    input: Option<String>,
    stack: Vec<String>,
    error: Option<String>,
}
use std::error::Error;

const PROMPT: &str = ">> ";

impl CalcDisplay for ReadlineCalcUI {
    fn init() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let editor = DefaultEditor::new()?;

        Ok(Self {
            radix: Radix::default(),
            rational: true,
            input: None,
            error: None,
            stack: vec![],
            editor,
        })
    }

    /// Wait on next Message for event loop
    fn next(&mut self) -> Option<Message> {
        loop {
            if let Some(line) = self.input.clone() {
                self.input = None;
                return Some(Message::Input(line));
            }

            if let Some(err) = self.error.as_ref() {
                println!("Error: {err}");
            } else {
                println!();
            }
            if self.stack.is_empty() {
                println!("   1:")
            } else {
                let len = self.stack.len();
                for (i, v) in self.stack.iter().enumerate() {
                    v.lines().for_each(|s| {
                        println!("{:>4}: {s}", len - i);
                    });
                }
            }
            let readline = self.editor.readline(PROMPT);

            match readline {
                Ok(line) => return Some(Message::Input(line)),
                Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
                Err(ReadlineError::WindowResized) => {} // redraw
                Err(e) => {
                    self.set_error(Some(e.to_string()));
                    continue;
                }
            }
        }
        None
    }

    /// Push input around again
    fn push_input(&mut self, value: String) {
        self.input = if value.is_empty() { None } else { Some(value) };
    }

    /// Set/Clear error message
    fn set_error(&mut self, msg: Option<String>) {
        self.error = msg;
    }

    /// Display Dialog with text
    fn dialog(&self, msg: String) {
        println!("{msg}");
    }

    fn set_data(&mut self, newdata: &[Value]) {
        let len = newdata.len();
        self.stack = newdata
            .iter()
            .enumerate()
            .map(|(i, v)| v.to_string_radix(self.radix, self.rational, i != len - 1))
            .collect();
    }

    fn set_display(&mut self, radix: Option<Radix>, rational: Option<bool>) {
        if let Some(rdx) = radix {
            self.radix = rdx;
        }
        if let Some(rational) = rational {
            self.rational = rational;
        }
    }

    /// Show Help Text
    fn help(&mut self) {
        println!("@@");
    }

    /// Cleanup and quit
    fn quit(&mut self) {}
}
