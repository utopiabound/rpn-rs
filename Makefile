prefix=${HOME}/.local
BINDIR=${HOME}/bin
ICONDIR=${prefix}/share/icons/hicolor/16x16/apps/


all: target/release/rpn-rs

install: target/release/rpn-rs
	install -m 0755 $< ${BINDIR}/
	install -m 0644 desktop/rpn-rs.png ${ICONDIR}/

target/release/rpn-rs: src/*.rs src/fixtures/help.html Cargo.*
	cargo build --release


