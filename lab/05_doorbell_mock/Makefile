CC   := gcc
CFLAGS := -O3 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -march=native -lpthread
TARGET := bench
ARGS ?=

.PHONY: all clean run

all: $(TARGET)

$(TARGET): bench.c
	$(CC) $(CFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)
