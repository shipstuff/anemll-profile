PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin

CC = xcrun clang
CFLAGS = -O2 -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML
SRC = anemll_profile.m
BIN = anemll-profile

.PHONY: all clean install uninstall

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

install: $(BIN)
	install -d $(BINDIR)
	install -m 755 $(BIN) $(BINDIR)/$(BIN)

uninstall:
	rm -f $(BINDIR)/$(BIN)

clean:
	rm -f $(BIN)
