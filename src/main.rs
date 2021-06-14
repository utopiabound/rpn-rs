use fltk::{
    app,
    enums::{Align, CallbackTrigger, Color, FrameType},
    group::Pack,
    input::Input,
    output::Output,
    prelude::{DisplayExt, GroupExt, InputExt, WidgetBase, WidgetExt},
    text::{TextBuffer, TextDisplay},
    window::Window,
};
use std::{collections::VecDeque, convert::TryFrom};

mod numbers;

use numbers::Value;

#[derive(Debug, Clone)]
enum Message {
    Input,
    //    Number(i32),
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

    let mut stacks: VecDeque<Vec<Value>> = VecDeque::with_capacity(stack_undo);
    stacks.push_front(vec![]);

    let width = 400;
    let out_h = 480;
    let in_h = 20;
    let err_h = 20;
    let win_h = out_h + in_h + err_h;

    let mut wind = Window::default().with_label("grpn").with_size(width, win_h);

    let pack = Pack::default().with_size(width, win_h);

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

    pack.end();

    app::set_focus(&input);
    //wind.make_resizable(true);
    wind.end();
    wind.show();

    let (s, r) = app::channel::<Message>();

    input.set_trigger(CallbackTrigger::EnterKey);
    input.emit(s, Message::Input);

    while app.wait() {
        if let Some(val) = r.recv() {
            match val {
                Message::Input => {
                    error.set_value("");
                    stacks.push_front(stacks[0].clone());

                    let rv = match input.value().as_str() {
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
                        "undo" | "u" => {
                            stacks.pop_front();
                            stacks.pop_front();
                            Return::Ok
                        }
                        "q" | "quit" => {
                            app::quit();
                            Return::Noop
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
                            let mut output = output_td.buffer().unwrap();
                            println!("writing: {:?}", stacks[0]);
                            output.set_text(
                                &stacks[0]
                                    .iter()
                                    .rev()
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            );
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
            }
        }
    }
}
