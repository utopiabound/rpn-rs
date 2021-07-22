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
    app,
    enums::{CallbackTrigger, Shortcut},
    group::Pack,
    input::Input,
    menu::{MenuFlag, SysMenuBar},
    output::Output,
    prelude::{GroupExt, InputExt, MenuExt, WidgetExt},
    window::Window,
};
use std::collections::VecDeque;
use std::convert::TryFrom;

#[derive(Debug, Copy, Clone)]
enum Message {
    Clear,
    Input,
    Radix(Radix),
    Quit,
}

fn main() {
    let stack_undo = 10;

    let app = app::App::default().with_scheme(app::Scheme::Gtk);
    app::set_visible_focus(false);
    app::background(0x42, 0x42, 0x42);
    app::background2(0x1b, 0x1b, 0x1b);

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
    if let Some(mut item) = menu.find_item("Radix/Decimal\t") {
        item.set();
    }

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

    input.set_trigger(CallbackTrigger::EnterKey);
    input.emit(s, Message::Input);

    while app.wait() {
        if let Some(val) = r.recv() {
            error.set_value("");
            let mut need_redisplay = false;
            match val {
                Message::Input => {
                    stacks.push_front(stacks[0].clone());

                    let rv = match input.value().as_str() {
                        "q" | "quit" => {
                            app::quit();
                            Return::Noop
                        }
                        // Stack Operations
                        "undo" | "u" => {
                            stacks.pop_front();
                            stacks.pop_front();
                            Return::Ok
                        }
                        "clear" => {
                            stacks.clear();
                            stacks.push_front(vec![]);
                            Return::Ok
                        }
                        "drop" | "pop" | "del" => {
                            if stacks[0].pop().is_some() {
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "sw" | "swap" => stacks[0].binary2(|a, b| (b, a)),
                        "dup" | "" => stacks[0].unary2(|a| (a.clone(), a)),
                        // Arithmatic Operations
                        "mod" => stacks[0].try_binary(|a, b| a.try_modulo(&b)),
                        "sqrt" => stacks[0].unary(|a| a.sqrt()),
                        "sqr" => stacks[0].unary(|a| a.clone() * a),
                        "pow" => stacks[0].try_binary(|a, b| a.try_pow(b)),
                        "+" => stacks[0].binary(|a, b| a + b),
                        "*" => stacks[0].binary(|a, b| a * b),
                        "-" => stacks[0].binary(|a, b| a - b),
                        "/" => stacks[0].binary(|a, b| a / b),
                        "<<" => stacks[0].try_binary(|a, b| a.try_lshift(&b)),
                        ">>" => stacks[0].try_binary(|a, b| a.try_rshift(&b)),
                        v => match Value::try_from(v) {
                            Ok(v) => {
                                stacks[0].push(v);
                                Return::Ok
                            }
                            Err(e) => Return::Err(e),
                        },
                    };

                    input.set_value("");

                    match rv {
                        Return::Ok => {
                            if stacks.len() == stack_undo {
                                stacks.pop_back();
                            }
                            need_redisplay = true;
                            println!("writing: {:?}", stacks[0]);
                        }
                        Return::Noop => {
                            stacks.pop_front();
                        }
                        Return::Err(e) => {
                            stacks.pop_front();
                            error.set_value(&e);
                        }
                    }

                    //println!("Max rows: {}", output.height()/output.text_size())
                }
                Message::Radix(r) => {
                    table.set_radix(r);
                    need_redisplay = true;
                }
                Message::Clear => {
                    stacks.clear();
                    stacks.push_front(vec![]);
                    need_redisplay = true;
                }
                Message::Quit => app::quit(),
            }
            if need_redisplay {
                table.set_data(&stacks[0]);
                table.redraw();
            }
        }
    }
}
