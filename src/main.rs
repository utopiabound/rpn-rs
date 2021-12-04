/* RPN-rs (c) 2021 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use rpn_rs::{
    display::StackOutput,
    numbers::{Radix, Value},
    stack::{Return, StackOps},
};

use fltk::{
    app, dialog,
    enums::{CallbackTrigger, Shortcut},
    group::Pack,
    input::Input,
    menu::{MenuFlag, SysMenuBar},
    output::Output,
    prelude::{GroupExt, InputExt, MenuExt, WidgetExt},
    window::Window,
};
use num_traits::Inv;
use rug::ops::Pow;
use std::collections::VecDeque;

#[derive(Debug, Copy, Clone)]
enum Message {
    About,
    Clear,
    Drop,
    Help,
    Input,
    Radix(Radix),
    Rational,
    Quit,
}

fn main() {
    let stack_undo = 10;

    env_logger::init();

    let app = app::App::default().with_scheme(app::Scheme::Gtk);
    app::set_visible_focus(false);
    //app::background(0x42, 0x42, 0x42);
    //app::background2(0x1b, 0x1b, 0x1b);

    let (s, r) = app::channel::<Message>();

    let mut stacks: VecDeque<Vec<Value>> = VecDeque::with_capacity(stack_undo);
    stacks.push_front(vec![]);

    let width = 400;
    let out_h = 480;
    let in_h = 20;
    let err_h = 20;
    let win_h = out_h + in_h + in_h + err_h;

    let mut wind = Window::default()
        .with_label("rpn-rs")
        .with_size(width, win_h);

    let pack = Pack::default().with_size(width, win_h);

    let mut menu = SysMenuBar::default().with_size(width, in_h);
    menu.add_emit(
        "File/Clear\t",
        Shortcut::None,
        MenuFlag::Normal,
        s,
        Message::Clear,
    );
    menu.add_emit(
        "File/Pop Stack\t",
        Shortcut::Ctrl | 'p',
        MenuFlag::Normal,
        s,
        Message::Drop,
    );
    menu.add_emit(
        "File/Quit\t",
        Shortcut::Ctrl | 'q',
        MenuFlag::Normal,
        s,
        Message::Quit,
    );
    menu.add_emit(
        "Radix/Decimal\t",
        Shortcut::Ctrl | 'd',
        MenuFlag::Radio,
        s,
        Message::Radix(Radix::Decimal),
    );
    menu.add_emit(
        "Radix/Hexadecimal\t",
        Shortcut::Ctrl | 'x',
        MenuFlag::Radio,
        s,
        Message::Radix(Radix::Hex),
    );
    menu.add_emit(
        "Radix/Octal\t",
        Shortcut::Ctrl | 'o',
        MenuFlag::Radio,
        s,
        Message::Radix(Radix::Octal),
    );
    menu.add_emit(
        "Radix/Binary\t",
        Shortcut::Ctrl | 'b',
        MenuFlag::Radio,
        s,
        Message::Radix(Radix::Binary),
    );
    menu.add_emit(
        "Options/Rational\t",
        Shortcut::Ctrl | 'r',
        MenuFlag::Toggle,
        s,
        Message::Rational,
    );
    menu.add_emit(
        "Help/About\t",
        Shortcut::None,
        MenuFlag::Normal,
        s,
        Message::About,
    );
    menu.add_emit(
        "Help/Help\t",
        Shortcut::None,
        MenuFlag::Normal,
        s,
        Message::Help,
    );
    menu.find_item("Radix/Decimal\t")
        .expect("Failed to find Decimal Radix")
        .set();
    menu.find_item("Options/Rational\t")
        .expect("Failed to find Rational Option")
        .set();

    let mut error = Output::default().with_size(width, err_h);

    let mut table = StackOutput::new(width, out_h);

    let mut input = Input::default().with_size(width, in_h);
    pack.resizable(table.table());

    pack.end();

    app::set_focus(&input);
    wind.make_resizable(true);
    wind.resizable(&pack);
    wind.end();
    wind.show();

    input.set_trigger(CallbackTrigger::EnterKeyAlways);
    input.emit(s, Message::Input);

    let mut help = dialog::HelpDialog::default();
    help.set_value(include_str!("fixtures/help.html"));
    help.hide();

    while app.wait() {
        if let Some(val) = r.recv() {
            error.set_value("");
            let mut need_redisplay = false;
            match val {
                Message::Input => {
                    stacks.push_front(stacks[0].clone());
                    let value = input.value();
                    input.set_value("");

                    let rv = match value.as_str() {
                        "q" | "quit" => {
                            app::quit();
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
                        "sw" | "swap" => stacks[0].binary_v(|a, b| vec![b, a]),
                        "dup" | "" => stacks[0].unary_v(|a| vec![a.clone(), a]),
                        // Arithmatic Operations
                        "+" => stacks[0].try_binary(|a, b| a + b),
                        "*" => stacks[0].try_binary(|a, b| a * b),
                        "-" => stacks[0].try_binary(|a, b| a - b),
                        "/" => stacks[0].try_binary(|a, b| a / b),
                        "inv" => stacks[0].try_unary(|a| a.inv()),
                        "ln" => stacks[0].try_unary(|a| a.try_ln()),
                        "log" => stacks[0].try_unary(|a| a.try_log10()),
                        "mod" => stacks[0].try_binary(|a, b| a.try_modulo(&b)),
                        "rem" | "%" => stacks[0].try_binary(|a, b| a % b),
                        "sqr" => stacks[0].try_unary(|a| a.clone() * a),
                        "sqrt" => stacks[0].unary(|a| a.sqrt()),
                        "^" | "pow" => stacks[0].try_binary(|a, b| a.pow(b)),
                        "dms" => stacks[0].try_unary(|a| a.try_dms_conv()),
                        "root" => stacks[0].try_binary(|a, b| a.try_root(b)),
                        "factor" => stacks[0].try_unary_v(|a| a.try_factor()),
                        // Matrix Operations
                        "det" | "determinant" => stacks[0].try_unary(|a| a.try_det()),
                        "trans" | "transpose" => stacks[0].try_unary(|a| a.try_transpose()),
                        "ident" | "identity" => stacks[0].try_unary(Value::identity),
                        // Binary Operations
                        "<<" => stacks[0].try_binary(|a, b| a.try_lshift(&b)),
                        ">>" => stacks[0].try_binary(|a, b| a.try_rshift(&b)),
                        // or, and, xor
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
                            let (v, op) = if v.ends_with(&['*', '/', '+', '-'][..]) {
                                let (val, op) = v.split_at(v.len() - 1);

                                (val, Some(op))
                            } else {
                                (v, None)
                            };
                            match Value::try_from(v) {
                                Ok(v) => {
                                    stacks[0].push(v);
                                    if let Some(op) = op {
                                        input.set_value(op);
                                        s.send(Message::Input);
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
                            error.set_value(&e);
                        }
                    }
                }
                Message::Drop => need_redisplay = stacks[0].pop().is_some(),
                Message::Radix(r) => {
                    table.set_radix(r);
                    need_redisplay = true;
                }
                Message::Rational => {
                    let item = menu
                        .find_item("Options/Rational\t")
                        .expect("Failed to find Rational Option");
                    table.set_rational(item.value());
                    need_redisplay = true;
                }
                Message::Clear => {
                    stacks.clear();
                    stacks.push_front(vec![]);
                    need_redisplay = true;
                }
                Message::About => dialog::message_default(r#"RPN Calculator (c) 2021"#),
                Message::Help => help.show(),
                Message::Quit => app::quit(),
            }
            if need_redisplay {
                table.set_data(&stacks[0]);
                table.redraw();
            }
        }
    }
}
