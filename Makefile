prefix=${HOME}/.local
BINDIR=${HOME}/bin
ICONDIR=${prefix}/share/icons/hicolor/16x16/apps

all: target/release/rpn-rs desktop/rpn-rs.icns

install: target/release/rpn-rs
	install -m 0755 $< ${BINDIR}/
	mkdir -p ${ICONDIR}
	convert -scale 16x16 desktop/rpn-rs.png ${ICONDIR}/rpn-rs.png
	mkdir -p ${prefix}/share/pixmaps/
	install -m 0644 desktop/rpn-rs.png ${prefix}/share/pixmaps/
	mkdir -p ${prefix}/share/applications/
	install -m 0644 desktop/rpn-rs.desktop ${prefix}/share/applications/

target/release/rpn-rs: src/*.rs src/fixtures/help.html Cargo.*
	cargo build --release

%.icns: %.png
	png2icns $@ $<

rpn-rs.app: all
	mkdir -p rpn-rs.app/Contents/Resources/
	mkdir rpn-rs.app/Contents/MacOS/
	cp desktop/rpn-rs.icns rpn-rs.app/Contents/Resources/
	cp desktop/Info.plist rpn-rs.app/Contents/
	cp target/release/rpn-rs rpn-rs.app/Contents/MacOS/

clean:
	rm -rf rpn-rs.app/

test:
	cargo fmt --check
	cargo clippy --locked --all
	cargo test --locked
	cargo spellcheck
