// Basically a table where the cell contents can be modified

use crate::numbers::{Radix, Value};
use clipboard::ClipboardProvider;
use fltk::{
    draw, enums,
    prelude::{GroupExt, TableExt, WidgetBase, WidgetExt},
    table,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct StackOutput {
    table: table::Table,
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

        table.set_rows(row_count(height, 1));
        table.set_row_header(true);
        table.set_row_resize(false);
        table.set_cols(2);
        table.set_col_header(false);
        table.set_col_width(0, width - 12 - (table.row_header_width() + 4));
        table.set_col_width(1, 10);
        table.set_col_resize(false);
        table.set_row_height_all(ROW_HEIGHT);
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
            e => {
                log::trace!("Table Event: {e:?}");
                false
            }
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
            table,
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
        let th = self.table.height();
        let row_count = row_count(th, v);
        self.table.set_rows(row_count);
        let h = v * ROW_HEIGHT;
        self.table.set_row_height_all(ROW_HEIGHT);
        self.table.set_row_height(row_count - 1, h);
        self.table.redraw()
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

    pub fn get_selection(&self) {
        let (x1, _, _, _) = self.table.get_selection();
        if x1 < 0 {
            log::debug!("No Selection");
            return;
        }
        let total_rows = self.table.rows();
        let len = self.data.borrow().len() as i32;
        let rn = len + x1 - total_rows;
        if rn < 0 {
            log::debug!("Selection is Empty");
            return;
        }

        let clip: Result<clipboard::ClipboardContext, _> = ClipboardProvider::new();

        if let Ok(mut clip) = clip {
            if let Some(v) = self.data.borrow().get(rn as usize) {
                let s = v.to_string_radix(
                    *self.radix.borrow(),
                    *self.rational.borrow(),
                    true,
                );
                log::debug!("Selection: {s}");
                if let Err(e) = clip.set_contents(s) {
                    log::error!("Failed clipboard set: {e:?}");
                }
            }
        }
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

    pub fn table(&self) -> &table::Table {
        &self.table
    }
}
