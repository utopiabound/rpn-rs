/* RPN-rs (c) 2024 Nathaniel Clark
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
use std::{collections::VecDeque, ops::Neg as _};

#[derive(Parser, Debug)]
/// RPN Calculator
struct App {
    #[clap(short, long, short_alias = 'u', default_value_t)]
    /// Type of `UI` to display
    flavor: Flavor,
}

fn handle_err(flavor: Flavor, e: impl std::fmt::Display) -> ! {
    eprintln!("Could not start rpn-rs in {flavor} mode: {e}");
    std::process::exit(1);
}

fn main() {
    let stack_undo = 10;

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
                    "quit" | "q" | "exit" | "#quit" | "#exit" => {
                        ui.quit();
                        break;
                    }
                    "help" | "?" | "#help" => {
                        ui.help();
                        Return::Noop
                    }
                    "about" | "#about" => {
                        ui.about();
                        Return::Noop
                    }
                    // Stack Operations
                    "undo" | "u" | "#undo" => {
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
                    "drop" | "del" | "delete" => {
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
                    "#real" | "#R" => {
                        ui.set_display(None, Some(false));
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#rational" | "#rat" | "#Q" => {
                        ui.set_display(None, Some(true));
                        need_redisplay = true;
                        Return::Noop
                    }
                    "#debug" => {
                        if let Some(val) = stacks[0].last() {
                            ui.dialog(format!("{val:?}"));
                        }
                        Return::Noop
                    }
                    // Arithmatic Operations
                    "+" => stacks[0].try_binary(|a, b| a + b),
                    "-" => stacks[0].try_binary(|a, b| a - b),
                    "*" => stacks[0].try_binary(|a, b| a * b),
                    "/" => stacks[0].try_binary(|a, b| a / b),
                    "!" => stacks[0].try_unary(|a| a.try_factorial()),
                    "abs" | "||" => stacks[0].try_unary(|a| a.try_abs()),
                    "exp" => stacks[0].try_unary(|a| a.try_exp()),
                    "inv" => stacks[0].try_unary(|a| a.inv()),
                    "ln" => stacks[0].try_unary(|a| a.try_ln()),
                    "log" | "log10" => stacks[0].try_unary(|a| a.try_log10()),
                    "log2" => stacks[0].try_unary(|a| a.try_log2()),
                    "neg" => stacks[0].unary(|a| a.neg()),
                    "mod" => stacks[0].try_binary(|a, b| a.try_modulo(&b)),
                    "%" | "rem" => stacks[0].try_binary(|a, b| a % b),
                    "sqr" => stacks[0].try_unary(|a| a.clone() * a),
                    "sqrt" => stacks[0].try_unary(|a| a.try_sqrt()),
                    "^" | "pow" => stacks[0].try_binary(|a, b| a.pow(b)),
                    "dms" => stacks[0].try_unary(|a| a.try_dms_conv()),
                    "permute" | "nPm" => stacks[0].try_binary(|a, b| a.try_permute(b)),
                    "choose" | "nCm" => stacks[0].try_binary(|a, b| a.try_choose(b)),
                    "root" => stacks[0].try_binary(|a, b| a.try_root(b)),
                    "round" | "rnd" => stacks[0].unary(|a| a.round()),
                    "trunc" | "truncate" => stacks[0].unary(|a| a.trunc()),
                    "floor" => stacks[0].unary(|a| a.floor()),
                    "ceil" => stacks[0].unary(|a| a.ceil()),
                    "factor" => stacks[0].try_unary(|a| a.try_factor()),
                    // Tuple Operations
                    "collect" => stacks[0].try_reduce(|acc, e| acc.try_push(e)),
                    "expand" => stacks[0].try_unary_v(|a| a.try_expand()),
                    "push" => stacks[0].try_binary(|a, b| b.try_push(a)),
                    "unpush" => stacks[0].try_unary_v(|a| a.try_unpush()),
                    "pop" => stacks[0].try_unary_v(|a| a.try_pull()),
                    // Tuple / Stats Operations
                    "avg" | "mean" => stacks[0].unary(|a| a.mean()),
                    "median" => stacks[0].unary(|a| a.median()),
                    "sort" => stacks[0].unary(|a| a.sort()),
                    "sum" => stacks[0].unary(|a| a.sum()),
                    "prod" | "product" => stacks[0].unary(|a| a.product()),
                    "sdev" | "sigma" => stacks[0].unary(|a| a.standard_deviation()),

                    // Matrix Operations
                    "det" | "determinant" => stacks[0].try_unary(|a| a.try_det()),
                    "trans" | "transpose" => stacks[0].try_unary(|a| a.try_transpose()),
                    "I" | "ident" | "identity" => stacks[0].try_unary(Value::identity),
                    "J" | "ones" => stacks[0].try_unary(Value::ones),
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
                    "pi" => {
                        stacks[0].push(Value::pi());
                        Return::Ok
                    }
                    "G" | "catalan" => {
                        stacks[0].push(Value::catalan());
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
