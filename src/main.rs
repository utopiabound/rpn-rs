/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

mod numbers;
mod stack;
mod ui;

use crate::{
    numbers::{Radix, Value},
    stack::{Return, StackOps},
    ui::{Flavor, Message},
};
use clap::Parser;
use num_traits::Inv;
use rug::ops::Pow;
use std::collections::VecDeque;

#[derive(Parser, Debug)]
/// RPN Calculator
struct App {
    #[clap(short, long, default_value_t)]
    /// Type of UI to display
    flavor: Flavor,
}

fn handle_err(flavor: Flavor, e: impl std::fmt::Display) -> ! {
    eprintln!("Could not start rpn-rs in {flavor} mode: {e}");
    std::process::exit(1);
}

fn main() {
    let stack_undo = 10;

    env_logger::init();

    let cmd = App::parse();

    let mut ui = match ui::get_ui(cmd.flavor) {
        Ok(x) => x,
        Err(e) => {
            if cmd.flavor == Flavor::Gui {
                let Ok(ui) = ui::get_ui(Flavor::Cli) else {
                    handle_err(Flavor::Cli, e);
                };
                ui
            } else {
                handle_err(cmd.flavor, e);
            }
        }
    };

    let mut stacks: VecDeque<Vec<Value>> = VecDeque::with_capacity(stack_undo);
    stacks.push_front(vec![]);

    while let Some(msg) = ui.next() {
        ui.set_error(None);

        let mut need_redisplay = false;
        match msg {
            Message::Input(value) => {
                stacks.push_front(stacks[0].clone());

                let rv = match value.as_str() {
                    "q" | "quit" => {
                        ui.quit();
                        break;
                    }
                    "help" | "?" => {
                        ui.help();
                        Return::Noop
                    }
                    // Stack Operations
                    "undo" | "u" => {
                        if stacks.len() > 2 {
                            stacks.pop_front();
                            stacks.pop_front();
                            Return::Ok
                        } else {
                            Return::Err("No further undos in buffer".to_string())
                        }
                    }
                    "clear" => {
                        stacks.clear();
                        stacks.push_front(vec![]);
                        Return::Ok
                    }
                    "drop" | "pop" | "del" | "delete" => {
                        if stacks[0].pop().is_some() {
                            Return::Ok
                        } else {
                            Return::Noop
                        }
                    }
                    "roll" | "rollu" | "ru" | "rup" | "rollup" => match stacks[0].pop() {
                        Some(x) => {
                            stacks[0].insert(0, x);
                            Return::Ok
                        }
                        None => Return::Noop,
                    },
                    "rolld" | "rd" | "rdown" | "rolldown" => {
                        if stacks[0].is_empty() {
                            Return::Noop
                        } else {
                            let x = stacks[0].remove(0);
                            stacks[0].push(x);
                            Return::Ok
                        }
                    }
                    "sw" | "swap" => stacks[0].binary_v(|a, b| vec![b, a]),
                    "dup" | "" => stacks[0].unary_v(|a| vec![a.clone(), a]),
                    // Display Operations
                    "#bin" => {
                        ui.set_display(Some(Radix::Binary), None);
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#oct" => {
                        ui.set_display(Some(Radix::Octal), None);
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#dec" => {
                        ui.set_display(Some(Radix::Decimal), None);
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#hex" => {
                        ui.set_display(Some(Radix::Hex), None);
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#real" => {
                        ui.set_display(None, Some(false));
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#rational" | "#rat" => {
                        ui.set_display(None, Some(true));
                        need_redisplay = true;
                        Return::Noop
                    }
                    // Arithmatic Operations
                    "+" => stacks[0].try_binary(|a, b| a + b),
                    "*" => stacks[0].try_binary(|a, b| a * b),
                    "-" => stacks[0].try_binary(|a, b| a - b),
                    "/" => stacks[0].try_binary(|a, b| a / b),
                    "!" => stacks[0].try_unary(|a| a.try_factorial()),
                    "inv" => stacks[0].try_unary(|a| a.inv()),
                    "ln" => stacks[0].try_unary(|a| a.try_ln()),
                    "log" => stacks[0].try_unary(|a| a.try_log10()),
                    "mod" => stacks[0].try_binary(|a, b| a.try_modulo(&b)),
                    "%" | "rem" => stacks[0].try_binary(|a, b| a % b),
                    "sqr" => stacks[0].try_unary(|a| a.clone() * a),
                    "sqrt" => stacks[0].try_unary(|a| a.try_sqrt()),
                    "^" | "pow" => stacks[0].try_binary(|a, b| a.pow(b)),
                    "dms" => stacks[0].try_unary(|a| a.try_dms_conv()),
                    "root" => stacks[0].try_binary(|a, b| a.try_root(b)),
                    "trunc" | "truncate" => stacks[0].try_unary(|a| a.try_trunc()),
                    "factor" => stacks[0].try_unary_v(|a| a.try_factor()),
                    // Matrix Operations
                    "det" | "determinant" => stacks[0].try_unary(|a| a.try_det()),
                    "trans" | "transpose" => stacks[0].try_unary(|a| a.try_transpose()),
                    "ident" | "identity" => stacks[0].try_unary(Value::identity),
                    "rref" => stacks[0].try_unary(|a| a.try_rref()),
                    // Binary Operations
                    "<<" | "lshift" => stacks[0].try_binary(|a, b| a.try_lshift(&b)),
                    ">>" | "rshift" => stacks[0].try_binary(|a, b| a.try_rshift(&b)),
                    // or, and, xor, nand
                    // Constants
                    "e" => {
                        stacks[0].push(Value::e());
                        Return::Ok
                    }
                    "i" => {
                        stacks[0].push(Value::i());
                        Return::Ok
                    }
                    "pi" => {
                        stacks[0].push(Value::pi());
                        Return::Ok
                    }
                    v => {
                        let (v, op) = if v.ends_with(&['*', '/', '+', '-', '!', '%', '^'][..]) {
                            let (val, op) = v.split_at(v.len() - 1);

                            (val, Some(op))
                        } else {
                            (v, None)
                        };
                        match Value::try_from(v) {
                            Ok(v) => {
                                stacks[0].push(v);
                                if let Some(op) = op {
                                    ui.push_input(op.to_string());
                                }
                                Return::Ok
                            }
                            Err(e) => Return::Err(e),
                        }
                    }
                };

                match rv {
                    Return::Ok => {
                        if stacks.len() == stack_undo {
                            stacks.pop_back();
                        }
                        need_redisplay = true;
                        log::debug!("writing: {:?}", stacks[0]);
                    }
                    Return::Noop => {
                        stacks.pop_front();
                    }
                    Return::Err(e) => {
                        stacks.pop_front();
                        ui.set_error(Some(e));
                    }
                }
            }
            Message::Drop => need_redisplay = stacks[0].pop().is_some(),
            Message::Clear => {
                stacks.clear();
                stacks.push_front(vec![]);
                need_redisplay = true;
            }
        }
        if need_redisplay {
            ui.set_data(&stacks[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn verify_cli() {
        use super::App;
        use clap::CommandFactory as _;

        App::command().debug_assert()
    }
}
