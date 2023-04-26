/* RPN-rs (c) 2023 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::{
    numbers::{Radix, Value},
    ui::{CalcDisplay, Message},
};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{error::Error, io::Stdout};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame, Terminal,
};

#[derive(Debug, Clone, Default)]
struct CalcInfo {
    retry: Option<String>,
    stack: Vec<String>,
    error: Option<String>,
    input: String,
    radix: Radix,
    rational: bool,
    help_popup: bool,
    state: TableState,
}

pub struct TuiCalcUI {
    info: CalcInfo,
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

/// helper function to create a centered rect using up certain percentage of the available rect `r`
// From tui-rs/examples/popup.rs
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(popup_layout[1])[1]
}

fn ui(info: &mut CalcInfo, f: &mut Frame<CrosstermBackend<Stdout>>) {
    let chunks = Layout::default()
        .constraints([Constraint::Min(1), Constraint::Length(1)].as_ref())
        .margin(0)
        .split(f.size());
    let height =
        chunks[0].height as usize - info.stack.get(0).map(|x| x.lines().count()).unwrap_or(1);

    let mut rows = (0..)
        .take(height)
        .map(|x| {
            let data = info
                .stack
                .get(x)
                .cloned()
                .unwrap_or_else(|| " ".to_string());
            let lines = data.lines().count() as u16;
            // @@/TODO: Right justify data Cell (c.f. https://github.com/fdehau/tui-rs/issues/295)
            Row::new(vec![Cell::from(format!("{:4}: ", x + 1)), Cell::from(data)]).height(lines)
        })
        .collect::<Vec<_>>();

    rows.reverse();

    let widths = [
        Constraint::Length(6),
        Constraint::Length(f.size().width - 7),
    ];
    let t = Table::new(rows)
        .block(Block::default().borders(Borders::NONE).title(Span::styled(
            info.error.clone().unwrap_or_default(),
            Style::default().fg(Color::Red),
        )))
        .widths(&widths);
    f.render_stateful_widget(t, chunks[0], &mut info.state);
    let input = Paragraph::new(info.input.as_ref());
    f.render_widget(input, chunks[1]);

    if info.help_popup {
        let block = Block::default().title("Help").borders(Borders::ALL);
        let area = centered_rect(80, 80, f.size());
        f.render_widget(block, area);
    } else {
        f.set_cursor(chunks[1].x + info.input.len() as u16, chunks[1].y);
    }
}

impl CalcDisplay for TuiCalcUI {
    fn init() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let stdout = std::io::stdout();

        enable_raw_mode()?;
        execute!(&stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            info: CalcInfo {
                rational: true,
                ..Default::default()
            },
            terminal,
        })
    }

    /// Cleanup and quit
    fn quit(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen,);
        let _ = self.terminal.show_cursor();
    }

    /// Wait on next Message for event loop
    fn next(&mut self) -> Option<Message> {
        loop {
            if let Err(e) = self.terminal.draw(|f| ui(&mut self.info, f)) {
                eprintln!("{e}");
                break;
            }

            if let Some(input) = self.info.retry.clone() {
                self.info.retry = None;
                return Some(Message::Input(input));
            }

            match event::read() {
                Err(e) => {
                    eprintln!("Error: {e}");
                    break;
                }
                Ok(Event::Key(k)) => {
                    if self.info.help_popup {
                        // @@
                        self.info.help_popup = false;
                    } else {
                        match k.code {
                            KeyCode::Char(c) if k.modifiers.is_empty() => self.info.input.push(c),
                            KeyCode::Enter => {
                                let line = self.info.input.clone();
                                self.info.input = "".to_string();
                                return Some(Message::Input(line));
                            }
                            KeyCode::Backspace => {
                                let _ = self.info.input.pop();
                            }
                            // TODO: Add line editing
                            _ => {}
                        }
                    }
                }
                Ok(Event::Paste(val)) => self.info.input.push_str(&val),
                Ok(Event::Resize(_, _)) => {}
                Ok(e) => self.set_error(Some(format!("Unknown: {e:?}"))),
            }
        }
        None
    }

    /// Push input around again
    fn push_input(&mut self, value: String) {
        self.info.retry = if value.is_empty() { None } else { Some(value) };
    }

    /// Set/Clear error message
    fn set_error(&mut self, msg: Option<String>) {
        self.info.error = msg;
    }

    /// Display Dialog with text
    fn dialog(&self, msg: String) {
        println!("{msg}");
    }

    fn set_data(&mut self, newdata: &[Value]) {
        let mut data = newdata.to_vec();
        data.reverse();
        self.info.stack = data
            .iter()
            .enumerate()
            .map(|(i, v)| v.to_string_radix(self.info.radix, self.info.rational, i != 0))
            .collect();
    }

    fn set_display(&mut self, radix: Option<Radix>, rational: Option<bool>) {
        if let Some(rdx) = radix {
            self.info.radix = rdx;
        }
        if let Some(rational) = rational {
            self.info.rational = rational;
        }
    }

    /// Show Help Text
    fn help(&mut self) {
        self.info.help_popup = true;
    }
}
