/* RPN-rs (c) 2025 Nathaniel Clark
 *
 * This source code is subject to the terms of the GPL v2. See LICENCE file.
 */

use crate::{
    numbers::{Radix, Value},
    ui::{about_txt, help_text, CalcDisplay, Info, Message},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, TableState},
    Frame, Terminal,
};
use std::{error::Error, io::Stdout};

#[derive(Debug, Clone, Default)]
struct CalcInfo {
    retry: Option<String>,
    stack: Vec<Value>,
    error: Option<String>,
    input: String,
    info: Info,
    help_popup: bool,
    // location of scroll in help text
    help_scroll: u16,
    // Maximum scroll value
    help_max_scroll: u16,
    // Height of help popup
    help_height: u16,
    state: TableState,
}

pub(crate) struct TuiCalcUI {
    info: CalcInfo,
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

/// helper function to create a centered rectangle using up certain percentage of the available rectangle `r`
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

fn ui(info: &mut CalcInfo, f: &mut Frame) {
    let chunks = Layout::default()
        .constraints([Constraint::Min(1), Constraint::Length(1)].as_ref())
        .margin(0)
        .split(f.area());
    let height = chunks[0].height as usize - info.stack.first().map(|x| x.lines()).unwrap_or(1);

    let data_width = f.area().width - 7;

    let mut rows = (0..)
        .take(height)
        .map(|x| {
            let data = info
                .stack
                .get(x)
                .map(|num| {
                    Text::from(num.to_string_radix(
                        info.info.radix,
                        info.info.rational,
                        x != 0,
                        (data_width - 1) as usize,
                    ))
                    .alignment(Alignment::Right)
                })
                .unwrap_or_else(|| Text::from(" "));
            let lines = data.lines.len() as u16;
            Row::new(vec![Cell::from(format!("{:4}:", x + 1)), Cell::from(data)]).height(lines)
        })
        .collect::<Vec<_>>();

    rows.reverse();

    let widths = [Constraint::Length(6), Constraint::Length(data_width)];
    let t = Table::new(rows, widths)
        .block(
            Block::default()
                .borders(Borders::NONE)
                .title(Line::from(vec![
                    Span::styled(format!("{:9}", info.info.radix), Modifier::BOLD),
                    Span::styled(format!("{:9}", info.info.angle), Modifier::BOLD),
                    Span::styled(
                        info.error.clone().unwrap_or_default(),
                        Style::default().fg(Color::Red),
                    ),
                ])),
        )
        .column_spacing(1);
    f.render_stateful_widget(t, chunks[0], &mut info.state);
    let input = Paragraph::new(info.input.as_str());
    f.render_widget(input, chunks[1]);

    if info.help_popup {
        let block = Block::default().title("Help").borders(Borders::ALL);
        let area = centered_rect(80, 80, f.area());
        let inner = block.inner(area);
        let text = help_text(inner.width as usize);
        info.help_height = inner.height;
        info.help_max_scroll = text
            .lines()
            .count()
            .checked_sub(inner.height.into())
            .unwrap_or_default() as u16;
        let p = Paragraph::new(text).scroll((info.help_scroll, 0));
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        f.render_widget(p, inner);
    } else {
        f.set_cursor_position(Position::new(
            chunks[1].x + info.input.len() as u16,
            chunks[1].y,
        ));
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
            info: CalcInfo::default(),
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
                    let kcode = match k.code {
                        KeyCode::Char(c) if k.modifiers.contains(KeyModifiers::SHIFT) => {
                            KeyCode::Char(c.to_ascii_uppercase())
                        }
                        _ => k.code,
                    };
                    // These match key bindings for less(1)
                    if self.info.help_popup {
                        match kcode {
                            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                                self.info.help_popup = false
                            }
                            KeyCode::Home | KeyCode::Char('g') | KeyCode::Char('p') => {
                                self.info.help_scroll = 0
                            }
                            KeyCode::Down
                            | KeyCode::Enter
                            | KeyCode::Char('e')
                            | KeyCode::Char('j')
                            | KeyCode::Char('J') => {
                                self.info.help_scroll = std::cmp::min(
                                    self.info.help_scroll + 1,
                                    self.info.help_max_scroll,
                                )
                            }
                            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('y') => {
                                self.info.help_scroll =
                                    self.info.help_scroll.checked_sub(1).unwrap_or_default()
                            }
                            KeyCode::Char('u') => {
                                self.info.help_scroll = self
                                    .info
                                    .help_scroll
                                    .checked_sub(self.info.help_height / 2)
                                    .unwrap_or_default()
                            }
                            KeyCode::PageDown
                            | KeyCode::Char(' ')
                            | KeyCode::Char('f')
                            | KeyCode::Char('z') => {
                                self.info.help_scroll = std::cmp::min(
                                    self.info.help_scroll + self.info.help_height,
                                    self.info.help_max_scroll,
                                )
                            }
                            KeyCode::Char('d') => {
                                self.info.help_scroll = std::cmp::min(
                                    self.info.help_scroll + self.info.help_height / 2,
                                    self.info.help_max_scroll,
                                )
                            }
                            KeyCode::PageUp | KeyCode::Char('b') | KeyCode::Char('w') => {
                                self.info.help_scroll = self
                                    .info
                                    .help_scroll
                                    .checked_sub(self.info.help_height)
                                    .unwrap_or_default()
                            }
                            KeyCode::End | KeyCode::Char('G') | KeyCode::Char('F') => {
                                self.info.help_scroll = self.info.help_max_scroll
                            }
                            _ => {}
                        }
                    } else {
                        match kcode {
                            KeyCode::Char(c) if k.modifiers.contains(KeyModifiers::CONTROL) => {
                                let mut info = self.get_info();
                                match c {
                                    'l' => {
                                        self.info.input = "".to_string();
                                        return Some(Message::Clear);
                                    }
                                    'p' => return Some(Message::Drop),
                                    'q' => return Some(Message::Input("quit".to_string())),
                                    'd' => info.radix = Radix::Decimal,
                                    'x' => info.radix = Radix::Hex,
                                    'b' => info.radix = Radix::Binary,
                                    'o' => info.radix = Radix::Octal,
                                    'r' => info.rational = !info.rational,
                                    'h' => return Some(Message::Input("help".to_string())),
                                    c => self.info.error = Some(format!("Unknown hotkey: ^{c}")),
                                }
                                self.set_info(info);
                            }
                            KeyCode::Char(c) => self.info.input.push(c),
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
    fn dialog(&mut self, msg: String) {
        self.set_error(Some(msg));
    }

    fn set_data(&mut self, newdata: &[Value]) {
        let mut data = newdata.to_vec();
        data.reverse();
        self.info.stack = data.to_vec();
    }

    fn set_info(&mut self, info: Info) {
        self.info.info = info;
    }

    fn get_info(&self) -> Info {
        self.info.info
    }

    fn about(&mut self) {
        self.info.error = Some(about_txt());
    }

    /// Show Help Text
    fn help(&mut self) {
        self.info.help_popup = true;
        self.info.help_scroll = 0;
    }
}
