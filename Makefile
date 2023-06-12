prefix=${HOME}/.local
BINDIR=${HOME}/bin
ICONDIR=${prefix}/share/icons/hicolor/16x16/apps/


all: target/release/rpn-rs

install: target/release/rpn-rs
	install -m 0755 $< ${BINDIR}/
	mkdir -p ${ICONDIR}
	install -m 0644 desktop/rpn-rs.png ${ICONDIR}/
	mkdir -p ${prefix}/share/pixmaps/
	install -m 0644 desktop/rpn-rs.png ${prefix}/share/pixmaps/
	mkdir -p ${prefix}/share/applications/
	install -m 0644 desktop/rpn-rs.desktop ${prefix}/share/applications/

target/release/rpn-rs: src/*.rs src/fixtures/help.html Cargo.*
	cargo build --release


test:
	cargo clippy --all
	cargo test
