/* RPN-rs (c) 2025 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::{
    numbers::Value,
    ui::{about_txt, help_text, CalcDisplay, Info, Message},
};

use rustyline::{
    error::{ReadlineError, Signal},
    DefaultEditor,
};

pub(crate) struct ReadlineCalcUI {
    info: Info,
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
            info: Info::default(),
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
                Err(ReadlineError::Eof)
                | Err(ReadlineError::Interrupted)
                | Err(ReadlineError::Signal(Signal::Interrupt)) => break,
                Err(ReadlineError::Signal(Signal::Resize)) => {} // redraw
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
    fn dialog(&mut self, msg: String) {
        println!("{msg}");
    }

    fn set_data(&mut self, newdata: &[Value]) {
        let len = newdata.len();
        self.stack = newdata
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.to_string_radix(self.info.radix, self.info.rational, i != len - 1, None)
            })
            .collect();
    }

    fn set_info(&mut self, info: Info) {
        self.info = info;
    }

    fn get_info(&self) -> Info {
        self.info
    }

    /// Show Help Text
    fn help(&mut self) {
        println!("{}", help_text(80));
    }

    fn about(&mut self) {
        println!("{}", about_txt());
    }

    /// Cleanup and quit
    fn quit(&mut self) {}
}
