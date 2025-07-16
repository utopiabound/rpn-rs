/* RPN-rs (c) 2025 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */
/*
 * A table where the cell contents can be modified
 */

use crate::{
    numbers::{Angle, Radix, Value},
    ui::{about_txt, CalcDisplay, Info, Message, HELP_HTML},
};
use copypasta::{ClipboardContext, ClipboardProvider};
use fltk::{
    app, dialog, draw,
    enums::{self, CallbackTrigger, Shortcut},
    group::Pack,
    image::PngImage,
    input::Input,
    menu::{MenuFlag, SysMenuBar},
    output::Output,
    prelude::{GroupExt, InputExt, MenuExt, TableExt, WidgetBase, WidgetExt, WindowExt},
    table,
    window::Window,
};
use std::{cell::RefCell, error::Error, rc::Rc};

#[derive(Debug, Copy, Clone)]
enum FltkMessage {
    About,
    Angle(Angle),
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

pub(crate) struct FltkCalcDisplay {
    app: fltk::app::App,
    table: StackOutput,
    angle: Angle,
    input: Input,
    error: Output,
    rx: app::Receiver<FltkMessage>,
    tx: app::Sender<FltkMessage>,
    help: dialog::HelpDialog,
    menu: SysMenuBar,
    clipboard: Box<dyn ClipboardProvider>,
}

const ICON: &[u8] = include_bytes!("../../desktop/rpn-rs.png");
const MENU_RATIONAL: &str = "Display/Rational\t";

trait MenuName {
    fn to_menu(&self) -> &str;
}

impl MenuName for Radix {
    fn to_menu(&self) -> &str {
        match self {
            Radix::Binary => "Display/Binary\t",
            Radix::Decimal => "Display/Decimal\t",
            Radix::Hex => "Display/Hexadecimal\t",
            Radix::Octal => "Display/Octal\t",
        }
    }
}

impl MenuName for Angle {
    fn to_menu(&self) -> &str {
        match self {
            Angle::Degree => "Option/Degrees\t",
            Angle::Radian => "Option/Radians\t",
            Angle::Gradian => "Option/Gradians\t",
        }
    }
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
            MENU_RATIONAL,
            Shortcut::Ctrl | 'r',
            MenuFlag::Toggle,
            tx,
            FltkMessage::Rational,
        );
        menu.add_emit(
            Radix::Decimal.to_menu(),
            Shortcut::Ctrl | 'd',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Decimal),
        );
        menu.add_emit(
            Radix::Hex.to_menu(),
            Shortcut::Ctrl | 'x',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Hex),
        );
        menu.add_emit(
            Radix::Octal.to_menu(),
            Shortcut::Ctrl | 'o',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Octal),
        );
        menu.add_emit(
            Radix::Binary.to_menu(),
            Shortcut::Ctrl | 'b',
            MenuFlag::Radio,
            tx,
            FltkMessage::Radix(Radix::Binary),
        );
        menu.add_emit(
            Angle::Degree.to_menu(),
            Shortcut::None,
            MenuFlag::Radio,
            tx,
            FltkMessage::Angle(Angle::Degree),
        );
        menu.add_emit(
            Angle::Radian.to_menu(),
            Shortcut::None,
            MenuFlag::Radio,
            tx,
            FltkMessage::Angle(Angle::Radian),
        );
        menu.add_emit(
            Angle::Gradian.to_menu(),
            Shortcut::None,
            MenuFlag::Radio,
            tx,
            FltkMessage::Angle(Angle::Gradian),
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

        let error = Output::default().with_size(width, err_h);

        let table = StackOutput::new(width, out_h);
        pack.resizable(&*table.table().borrow());

        // Set Menu Defaults
        menu.find_item(table.get_radix().to_menu())
            .expect("Failed to find Decimal Radix")
            .set();
        menu.find_item(MENU_RATIONAL)
            .expect("Failed to find Rational Option")
            .set();
        menu.find_item(Angle::default().to_menu())
            .expect("Failed to find Degrees Option")
            .set();

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
        wind.set_icon(PngImage::from_data(ICON).ok());

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
            angle: Angle::default(),
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
                FltkMessage::Angle(angle) => {
                    self.angle = angle;
                }
                FltkMessage::Rational => {
                    let item = self
                        .menu
                        .find_item(MENU_RATIONAL)
                        .expect("Failed to find Rational Option");
                    self.table.set_rational(item.value());
                    self.table.redraw();
                }
                FltkMessage::About => self.about(),
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

    fn set_info(&mut self, info: Info) {
        let old_info = self.get_info();
        if old_info.radix != info.radix {
            self.table.set_radix(info.radix);
            self.menu
                .find_item(old_info.radix.to_menu())
                .expect("failed to find old Radix menu")
                .clear();
            self.menu
                .find_item(info.radix.to_menu())
                .expect("Failed to find new Radix Menu")
                .set();
        }
        if old_info.rational != info.rational {
            self.table.set_rational(info.rational);
            let mut item = self
                .menu
                .find_item(MENU_RATIONAL)
                .expect("Failed to find Rational Option");
            if info.rational {
                item.set();
            } else {
                item.clear();
            }
        }
        // @@
        self.angle = info.angle;
    }

    fn get_info(&self) -> Info {
        Info {
            radix: self.table.get_radix(),
            rational: self.table.get_rational(),
            angle: self.angle,
        }
    }

    fn set_error(&mut self, err: Option<String>) {
        self.error.set_value(err.unwrap_or_default().as_str());
    }

    fn about(&mut self) {
        self.dialog(about_txt());
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
    pub(crate) fn new(width: i32, height: i32) -> Self {
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
        table.set_col_width(1, 10);
        table.set_col_resize(false);
        table.set_row_height_all(ROW_HEIGHT);
        table.make_resizable(true);
        let d2 = data.clone();
        table.resize_callback(move |s, _x, _y, width, height| {
            let first = d2.borrow().last().map(|x| x.lines() as i32).unwrap_or(1);
            let rc = row_count(height, first);
            s.set_rows(rc);
            let first_rh = first * ROW_HEIGHT;
            s.set_row_height_all(ROW_HEIGHT);
            s.set_row_height(rc - 1, first_rh);
            s.set_col_width(0, width - 12 - (s.row_header_width() + 4));
            s.redraw();
        });
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
                draw::set_font(enums::Font::Screen, 18);
                let digits = w as usize / draw::char_width('m') as usize;
                let value = if col != 0 || rn < 0 {
                    "".to_string()
                } else {
                    data_c.borrow()[rn as usize].to_string_radix(
                        *radix_c.borrow(),
                        *rational_c.borrow(),
                        row + 1 != total_rows,
                        digits - 2,
                    )
                };
                let selected = t.is_selected(row, col);
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
                draw::draw_text2(&value, x, y, w, h, enums::Align::Right);
                draw::pop_clip();
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

    pub(crate) fn redraw(&mut self) {
        let w = self.table.borrow().width();
        let h = self.table.borrow().height();
        // cause table resize_callback to be called
        self.table.borrow_mut().set_size(w, h);
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

    pub(crate) fn get_selection(&self) -> Result<String, String> {
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
            .map(|v| v.to_string_radix(*self.radix.borrow(), *self.rational.borrow(), true, None))
            .ok_or_else(|| "No Selection".to_string())
    }

    pub(crate) fn set_data(&mut self, newdata: &[Value]) {
        let mut data = self.data.borrow_mut();
        data.clear();
        data.extend_from_slice(newdata);
    }

    pub(crate) fn set_radix(&mut self, radix: Radix) {
        *self.radix.borrow_mut() = radix;
    }

    pub(crate) fn get_radix(&self) -> Radix {
        *self.radix.borrow()
    }

    pub(crate) fn set_rational(&mut self, rational: bool) {
        *self.rational.borrow_mut() = rational;
    }

    pub(crate) fn get_rational(&self) -> bool {
        *self.rational.borrow()
    }

    pub(crate) fn table(&self) -> Rc<RefCell<table::Table>> {
        self.table.clone()
    }
}
