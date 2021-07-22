// Basically a table where the cell contents can be modified

use crate::numbers::{Radix, Value};
use fltk::{
    draw, enums,
    prelude::{GroupExt, TableExt, WidgetBase, WidgetExt},
    table,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct StackOutput {
    table: table::Table,
    rows: Rc<RefCell<i32>>,
    radix: Rc<RefCell<Radix>>,
    rational: Rc<RefCell<bool>>,
    data: Rc<RefCell<Vec<Value>>>,
}

const ROW_HEIGHT: i32 = 28;

fn row_count(h: i32) -> i32 {
    1 + (h / ROW_HEIGHT)
}

impl StackOutput {
    pub fn new(width: i32, height: i32) -> Self {
        let mut table = table::Table::default()
            .with_size(width, height)
            .center_of_parent();
        let data: Rc<RefCell<Vec<Value>>> = Rc::from(RefCell::from(vec![]));
        let radix = Rc::from(RefCell::from(Radix::Decimal));
        let rational = Rc::from(RefCell::from(true));
        let rows = Rc::from(RefCell::from(row_count(height)));

        table.set_row_height_all(ROW_HEIGHT);
        table.set_rows(*rows.borrow());
        table.set_row_header(true);
        table.set_row_resize(false);
        table.set_cols(2);
        table.set_col_header(false);
        table.set_col_width(0, width - 10 - (table.row_header_width()+4));
        table.set_col_width(1, 10);
        table.set_col_resize(false);
        table.end();

        //println!("Table {}x{}", table.rows(), table.cols());

        let data_c = data.clone();
        let radix_c = radix.clone();
        let rational_c = rational.clone();
        let rows_c = rows.clone();

        // push focus from Text display to input
        table.handle(move |s, e| {
            match e {
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
                    let r = row_count(s.height());
                    if r != s.rows() {
                        *rows_c.borrow_mut() = r;
                        s.set_rows(r);
                        //println!("D: Set rows = {}", r);
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        });

        let rows_c = rows.clone();

        // Called when the table is drawn then when it's redrawn due to events
        table.draw_cell(move |t, ctx, row, col, x, y, w, h| match ctx {
            table::TableContext::StartPage => draw::set_font(enums::Font::Helvetica, 14),
            table::TableContext::RowHeader => {
                Self::draw_header(&format!("{}:", *rows_c.borrow() - row), x, y, w, h)
            } // Row titles
            table::TableContext::Cell => {
                let rn = (data_c.borrow().len() as i32) - *rows_c.borrow() + row;
                //println!("ROW:{} rows={} index={}", row,  *rows_c.borrow(), rn);
                let value = if col == 0 && rn >= 0 {
                    data_c.borrow()[rn as usize].to_string_radix(
                        *radix_c.borrow(),
                        *rational_c.borrow(),
                    )
                } else {
                    "".to_string()
                };
                Self::draw_data(
                    &value,
                    x,
                    y,
                    w,
                    h,
                    t.is_selected(row, col),
                );
            }
            _ => (),
        });

        Self { table, rows, data, radix, rational }
    }

    pub fn redraw(&mut self) {
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

    // The selected flag sets the color of the cell to a grayish color, otherwise white
    fn draw_data(txt: &str, x: i32, y: i32, w: i32, h: i32, selected: bool) {
        draw::push_clip(x, y, w, h);
        if selected {
            draw::set_draw_color(enums::Color::from_u32(0x00D3_D3D3));
        } else {
            draw::set_draw_color(enums::Color::BackGround);
        }
        draw::draw_rectf(x, y, w, h);
        draw::set_draw_color(enums::Color::Gray0);
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
