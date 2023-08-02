/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */
/*
 * A table where the cell contents can be modified
 */

use crate::{
    numbers::{Radix, Value},
    ui::{CalcDisplay, Message, HELP_HTML},
};
use copypasta::{ClipboardContext, ClipboardProvider};
use fltk::{
    app, dialog, draw,
    enums::{self, CallbackTrigger, Shortcut},
    group::Pack,
    input::Input,
    menu::{MenuFlag, SysMenuBar},
    output::Output,
    prelude::{GroupExt, InputExt, MenuExt, TableExt, WidgetBase, WidgetExt},
    table,
    window::Window,
};
use std::{cell::RefCell, error::Error, rc::Rc};

#[derive(Debug, Copy, Clone)]
enum FltkMessage {
    About,
    Clear,
    Drop,
    Copy,
    Paste,
    Help,
    Input,
    Radix(Radix),
    Rational,
    Quit,
}

pub struct FltkCalcDisplay {
    app: fltk::app::App,
    table: StackOutput,
    input: Input,
    error: Output,
    rx: app::Receiver<FltkMessage>,
    tx: app::Sender<FltkMessage>,
    help: dialog::HelpDialog,
    menu: SysMenuBar,
    clipboard: Box<dyn ClipboardProvider>,
}

fn resize_table(table: &mut table::Table, x: i32, y: i32, width: i32, height: i32) {
    table.set_rows(row_count(height, 1));
    table.set_col_width(0, width - 12 - (table.row_header_width() + 4));
}

impl CalcDisplay for FltkCalcDisplay {
    fn init() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let app = app::App::default().with_scheme(app::Scheme::Gtk);
        app::set_visible_focus(false);
        //app::background(0x42, 0x42, 0x42);
        //app::background2(0x1b, 0x1b, 0x1b);

        let (tx, rx) = app::channel::<FltkMessage>();
        let width = 400;
        let out_h = 480;
        let in_h = 20;
        let err_h = 20;
        let win_h = out_h + in_h + in_h + err_h;

        let mut wind = Window::default()
            .with_label("rpn-rs")
            .with_size(width, win_h);

        let mut pack = Pack::default().with_size(width, win_h);
        let clipboard = ClipboardContext::new()?;

        let mut menu = SysMenuBar::default().with_size(width, in_h);
        menu.add_emit(
            "File/Clear\t",
            Shortcut::None,
            MenuFlag::Normal,
            tx,
            FltkMessage::Clear,
        );
        menu.add_emit(
            "File/Pop Stack\t",
            Shortcut::Ctrl | 'p',
            MenuFlag::Normal,
            tx,
            FltkMessage::Drop,
        );
        menu.add_emit(
            "File/Copy\t",
            Shortcut::Ctrl | 'c',
            MenuFlag::Normal,
            tx,
            FltkMessage::Copy,
        );
        menu.add_emit(
            "File/Paste\t",
            Shortcut::Ctrl | 'v',
            MenuFlag::Normal,
            tx,
            FltkMessage::Paste,
        );
        menu.add_emit(
            "File/Quit\t",
            Shortcut::Ctrl | 'q',
            MenuFlag::Normal,
            tx,
            FltkMessage::Quit,
        );
        menu.add_emit(
            "Radix/Decimal\t",
            Shortcut::Ctrl | 'd',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Decimal),
        );
        menu.add_emit(
            "Radix/Hexadecimal\t",
            Shortcut::Ctrl | 'x',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Hex),
        );
        menu.add_emit(
            "Radix/Octal\t",
            Shortcut::Ctrl | 'o',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Octal),
        );
        menu.add_emit(
            "Radix/Binary\t",
            Shortcut::Ctrl | 'b',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Binary),
        );
        menu.add_emit(
            "Options/Rational\t",
            Shortcut::Ctrl | 'r',
            MenuFlag::Toggle,
            tx,
            FltkMessage::Rational,
        );
        menu.add_emit(
            "Help/About\t",
            Shortcut::None,
            MenuFlag::Normal,
            tx,
            FltkMessage::About,
        );
        menu.add_emit(
            "Help/Help\t",
            Shortcut::None,
            MenuFlag::Normal,
            tx,
            FltkMessage::Help,
        );
        menu.find_item("Radix/Decimal\t")
            .expect("Failed to find Decimal Radix")
            .set();
        menu.find_item("Options/Rational\t")
            .expect("Failed to find Rational Option")
            .set();

        let error = Output::default().with_size(width, err_h);

        let table = StackOutput::new(width, out_h);
        pack.resizable(&*table.table().borrow());

        let mut input = Input::default().with_size(width, in_h);

        let t2 = table.table();
        let other_height = win_h - out_h;
        pack.resize_callback(move |_s, _x, _y, w, h| {
            let tx = t2.borrow().x();
            let mut t = t2.borrow_mut();
            t.resize(0, tx, w, h - other_height);
        });
        pack.end();

        app::set_focus(&input);
        wind.make_resizable(true);
        wind.resizable(&pack);
        wind.end();
        wind.show();

        input.set_trigger(CallbackTrigger::EnterKeyAlways);
        input.emit(tx, FltkMessage::Input);

        let mut help = dialog::HelpDialog::default();
        help.set_value(HELP_HTML);
        help.hide();

        Ok(Self {
            table,
            app,
            input,
            error,
            help,
            rx,
            tx,
            menu,
            clipboard: Box::new(clipboard),
        })
    }

    fn next(&mut self) -> Option<Message> {
        while self.app.wait() {
            let Some(val) = self.rx.recv() else {
                continue;
            };
            match val {
                FltkMessage::Input => {
                    let v = Message::Input(self.input.value());
                    self.input.set_value("");
                    return Some(v);
                }
                FltkMessage::Clear => return Some(Message::Clear),
                FltkMessage::Drop => return Some(Message::Drop),
                FltkMessage::Radix(rdx) => {
                    self.table.set_radix(rdx);
                    self.table.redraw();
                }
                FltkMessage::Rational => {
                    let item = self
                        .menu
                        .find_item("Options/Rational\t")
                        .expect("Failed to find Rational Option");
                    self.table.set_rational(item.value());
                    self.table.redraw();
                }
                FltkMessage::About => self.dialog(format!(
                    "RPN Calculator {} (c) 2023",
                    env!("CARGO_PKG_VERSION")
                )),
                FltkMessage::Copy => {
                    if let Err(e) = self
                        .table
                        .get_selection()
                        .map(|txt| self.clipboard.set_contents(txt).map_err(|e| e.to_string()))
                    {
                        self.error.set_value(&e);
                    }
                }
                FltkMessage::Paste => match self.clipboard.get_contents() {
                    Ok(txt) => self.input.set_value(&txt),
                    Err(e) => self.error.set_value(&e.to_string()),
                },
                FltkMessage::Help => self.help(),
                FltkMessage::Quit => {
                    self.app.quit();
                    return None;
                }
            }
        }
        None
    }

    fn push_input(&mut self, value: String) {
        self.input.set_value(&value);
        self.tx.send(FltkMessage::Input);
    }

    fn set_data(&mut self, data: &[Value]) {
        self.table.set_data(data);
        self.table.redraw();
    }

    fn set_display(&mut self, rdx: Option<Radix>, rational: Option<bool>) {
        if let Some(rdx) = rdx {
            self.table.set_radix(rdx);
        }
        if let Some(rational) = rational {
            self.table.set_rational(rational);
        }
    }

    fn set_error(&mut self, err: Option<String>) {
        self.error.set_value(err.unwrap_or_default().as_str());
    }

    fn help(&mut self) {
        self.help.show();
    }

    fn dialog(&mut self, msg: String) {
        dialog::message_default(&msg);
    }

    /// Call back for Message::Quit
    fn quit(&mut self) {
        self.app.quit();
    }
}

struct StackOutput {
    table: Rc<RefCell<table::Table>>,
    radix: Rc<RefCell<Radix>>,
    rational: Rc<RefCell<bool>>,
    data: Rc<RefCell<Vec<Value>>>,
}

const ROW_HEIGHT: i32 = 26;

fn row_count(table_height: i32, lines: i32) -> i32 {
    (table_height / ROW_HEIGHT) + (1 - lines)
}

impl StackOutput {
    pub fn new(width: i32, height: i32) -> Self {
        let mut table = table::Table::default()
            .with_size(width, height)
            .center_of_parent();
        let data: Rc<RefCell<Vec<Value>>> = Rc::from(RefCell::from(vec![]));
        let radix = Rc::from(RefCell::from(Radix::Decimal));
        let rational = Rc::from(RefCell::from(true));

        table.set_row_header(true);
        //table.set_row_resize(false);
        table.set_cols(2);
        table.set_col_header(false);
        resize_table(&mut table, 0, 0, width, height);
        table.set_col_width(1, 10);
        table.set_col_resize(false);
        table.set_row_height_all(ROW_HEIGHT);
        table.make_resizable(true);
        table.resize_callback(resize_table);
        table.set_scrollbar_size(0);
        table.end();

        let data_c = data.clone();

        // push focus from Text display to input
        table.handle(move |s, e| match e {
            enums::Event::Focus => {
                if let Some(p) = s.parent() {
                    if let Some(mut c) = p.child(p.children() - 1) {
                        let _ = c.take_focus();
                        return true;
                    }
                }
                false
            }
            enums::Event::Resize => {
                let r = row_count(
                    s.height(),
                    data_c
                        .borrow()
                        .last()
                        .map(|x| x.lines() as i32)
                        .unwrap_or(1),
                );
                if r != s.rows() {
                    s.set_rows(r);
                }
                let w = s.width() - 12 - s.row_header_width();
                s.set_col_width(0, w);
                true
            }
            _ => false,
        });

        let data_c = data.clone();
        let radix_c = radix.clone();
        let rational_c = rational.clone();

        // Called when the table is drawn then when it's redrawn due to events
        table.draw_cell(move |t, ctx, row, col, x, y, w, h| match ctx {
            table::TableContext::StartPage => draw::set_font(enums::Font::Helvetica, 14),
            table::TableContext::RowHeader => {
                Self::draw_header(&format!("{}:", t.rows() - row), x, y, w, h)
            } // Row titles
            table::TableContext::Cell => {
                let total_rows = t.rows();
                let rn = (data_c.borrow().len() as i32) - total_rows + row;
                let value = if col != 0 || rn < 0 {
                    "".to_string()
                } else {
                    data_c.borrow()[rn as usize].to_string_radix(
                        *radix_c.borrow(),
                        *rational_c.borrow(),
                        row + 1 != total_rows,
                    )
                };
                Self::draw_data(&value, x, y, w, h, t.is_selected(row, col));
            }
            _ => (),
        });

        Self {
            table: Rc::from(RefCell::from(table)),
            radix,
            rational,
            data,
        }
    }

    pub fn redraw(&mut self) {
        let v = self
            .data
            .borrow()
            .last()
            .map(|x| x.lines() as i32)
            .unwrap_or(1);
        let mut table = self.table.borrow_mut();
        let th = table.height();
        let row_count = row_count(th, v);
        table.set_rows(row_count);
        let h = v * ROW_HEIGHT;
        table.set_row_height_all(ROW_HEIGHT);
        table.set_row_height(row_count - 1, h);
        table.redraw()
    }

    fn draw_header(txt: &str, x: i32, y: i32, w: i32, h: i32) {
        draw::push_clip(x, y, w, h);
        draw::draw_box(
            enums::FrameType::FlatBox,
            x,
            y,
            w,
            h,
            enums::Color::FrameDefault,
        );
        draw::set_draw_color(enums::Color::Black);
        draw::set_font(enums::Font::ScreenBold, 24);
        draw::draw_text2(txt, x, y, w, h, enums::Align::Right);
        draw::pop_clip();
    }

    pub fn get_selection(&self) -> Result<String, String> {
        let (x1, _, _, _) = self.table.borrow().get_selection();
        if x1 < 0 {
            return Err("No Selection".to_string());
        }
        let total_rows = self.table.borrow().rows();
        let len = self.data.borrow().len() as i32;
        let rn = len + x1 - total_rows;
        if rn < 0 {
            return Ok("".to_string());
        }

        self.data
            .borrow()
            .get(rn as usize)
            .map(|v| v.to_string_radix(*self.radix.borrow(), *self.rational.borrow(), true))
            .ok_or_else(|| "No Selection".to_string())
    }

    // The selected flag sets the color of the cell to a grayish color, otherwise white
    fn draw_data(txt: &str, x: i32, y: i32, w: i32, h: i32, selected: bool) {
        draw::push_clip(x, y, w, h);
        if selected {
            draw::set_draw_color(enums::Color::Selection);
        } else {
            draw::set_draw_color(enums::Color::BackGround);
        }
        draw::draw_rectf(x, y, w, h);
        if selected {
            draw::set_draw_color(enums::Color::White);
        } else {
            draw::set_draw_color(enums::Color::Gray0);
        }
        draw::set_font(enums::Font::Screen, 18);
        draw::draw_text2(txt, x, y, w, h, enums::Align::Right);
        //draw::draw_rect(x, y, w, h);
        draw::pop_clip();
    }

    pub fn set_data(&mut self, newdata: &[Value]) {
        let mut data = self.data.borrow_mut();
        data.clear();
        data.extend_from_slice(newdata);
    }

    pub fn set_radix(&mut self, radix: Radix) {
        *self.radix.borrow_mut() = radix;
    }

    pub fn set_rational(&mut self, rational: bool) {
        *self.rational.borrow_mut() = rational;
    }

    pub fn table(&self) -> Rc<RefCell<table::Table>> {
        self.table.clone()
    }
}
