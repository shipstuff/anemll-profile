PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin

CC = xcrun clang
SWIFTC = xcrun swiftc
CFLAGS = -O2 -fobjc-arc
SWIFTFLAGS = -O
FRAMEWORKS = -framework Foundation -framework CoreML
SRC = anemll_profile.m
BIN = anemll-profile

SRC_COSTPLAN = ane_costplan.swift
BIN_COSTPLAN = ane-costplan

.PHONY: all clean install uninstall

all: $(BIN) $(BIN_COSTPLAN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

$(BIN_COSTPLAN): $(SRC_COSTPLAN)
	$(SWIFTC) $(SWIFTFLAGS) -o $@ $< $(FRAMEWORKS) -parse-as-library

install: $(BIN) $(BIN_COSTPLAN)
	install -d $(BINDIR)
	install -m 755 $(BIN) $(BINDIR)/$(BIN)
	install -m 755 $(BIN_COSTPLAN) $(BINDIR)/$(BIN_COSTPLAN)

uninstall:
	rm -f $(BINDIR)/$(BIN)
	rm -f $(BINDIR)/$(BIN_COSTPLAN)

clean:
	rm -f $(BIN) $(BIN_COSTPLAN)
