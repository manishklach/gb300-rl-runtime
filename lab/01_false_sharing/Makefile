CC   := gcc
CFLAGS := -O3 -march=native -lpthread
TARGET := bench

.PHONY: all clean run

all: $(TARGET)

$(TARGET): bench.c
	$(CC) $(CFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
