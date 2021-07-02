/* RPN-rs (c) 2021 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use fltk::{
    app,
    enums::{Align, CallbackTrigger, Color, Event, FrameType, Shortcut},
    group::Pack,
    input::Input,
    menu::{MenuFlag, SysMenuBar},
    output::Output,
    prelude::{DisplayExt, GroupExt, InputExt, MenuExt, WidgetBase, WidgetExt},
    text::{TextBuffer, TextDisplay},
    window::Window,
};
use std::{collections::VecDeque, convert::TryFrom};

mod numbers;

use numbers::{Radix, Value};

#[derive(Debug, Copy, Clone)]
enum Message {
    Clear,
    Input,
    Radix(Radix),
    Quit,
}

enum Return {
    Ok,
    Noop,
    Err(String),
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

    let mut wind = Window::default().with_label("rpn-rs").with_size(width, win_h);

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

    let error = Output::default().with_size(width, err_h);

    let mut output_td = TextDisplay::default().with_size(width, out_h);
    let output = TextBuffer::default();
    output_td.set_buffer(output);
    output_td.set_align(Align::Right);
    output_td.set_frame(FrameType::FlatBox);
    output_td.set_text_color(Color::White);
    output_td.set_text_size(24);
    output_td.set_linenumber_width(30);

    let mut input = Input::default().with_size(width, in_h);

    // push focus from Text display to input
    output_td.handle(|s, e| {
        if e == Event::Focus {
            if let Some(p) = s.parent() {
                if let Some(mut c) = p.child(p.children()-1) {
                    let _ = c.take_focus();
                    return true;
                }
            }
        }
        false
    });

    pack.resizable(&output_td);

    pack.end();

    app::set_focus(&input);
    wind.make_resizable(true);
    wind.resizable(&pack);
    wind.end();
    wind.show();

    input.set_trigger(CallbackTrigger::EnterKey);
    input.emit(s, Message::Input);

    let mut radix = Radix::Decimal;
    let rational = true;

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
                            if stacks[0].len() > 0 {
                                stacks[0].pop();
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "sw" | "swap" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                stacks[0].push(a);
                                stacks[0].push(b);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        // Arithmatic Operations
                        "mod" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                match b.try_modulo(&a) {
                                    Ok(c) => {
                                        stacks[0].push(c);
                                        Return::Ok
                                    }
                                    Err(e) => Return::Err(e),
                                }
                            } else {
                                Return::Noop
                            }
                        }
                        "sqrt" => {
                            if let Some(a) = stacks[0].pop() {
                                let c = a.sqrt();
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "sqr" => {
                            if let Some(a) = stacks[0].pop() {
                                let c = a.clone() * a;
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "pow" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                match b.try_pow(a) {
                                    Ok(c) => {
                                        stacks[0].push(c);
                                        Return::Ok
                                    }
                                    Err(e) => Return::Err(e),
                                }
                            } else {
                                Return::Noop
                            }
                        }
                        "+" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                let c = b + a;
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "*" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                let c = b * a;
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "-" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                let c = b - a;
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "/" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                let c = b / a;
                                stacks[0].push(c);
                                Return::Ok
                            } else {
                                Return::Noop
                            }
                        }
                        "<<" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                match b.try_lshift(&a) {
                                    Ok(c) => {
                                        stacks[0].push(c);
                                        Return::Ok
                                    }
                                    Err(e) => Return::Err(e),
                                }
                            } else {
                                Return::Noop
                            }
                        }
                        ">>" => {
                            if stacks[0].len() > 1 {
                                let a = stacks[0].pop().unwrap();
                                let b = stacks[0].pop().unwrap();
                                match b.try_rshift(&a) {
                                    Ok(c) => {
                                        stacks[0].push(c);
                                        Return::Ok
                                    }
                                    Err(e) => Return::Err(e),
                                }
                            } else {
                                Return::Noop
                            }
                        }
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
                    if r != radix {
                        radix = r;
                        need_redisplay = true;
                    }
                }
                Message::Clear => {
                    stacks.clear();
                    stacks.push_front(vec![]);
                    need_redisplay = true;
                }
                Message::Quit => app::quit(),
            }
            if need_redisplay {
                let mut output = output_td.buffer().unwrap();
                output.set_text(
                    &stacks[0]
                        .iter()
                        .rev()
                        .map(|s| s.to_string_radix(radix, rational))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }
        }
    }
}
