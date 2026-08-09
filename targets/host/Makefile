# Host conformance build. Generate the sources first:
#   python examples/export_kangdn_c.py targets/host/generated
#   make -C targets/host run

CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Werror -pedantic
GEN     := generated

conformance: $(GEN)/kangdn.c main.c
	$(CC) $(CFLAGS) -I$(GEN) $(GEN)/kangdn.c main.c -o $@ -lm

.PHONY: run clean
run: conformance
	./conformance

clean:
	rm -f conformance
