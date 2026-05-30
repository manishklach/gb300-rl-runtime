CC   := gcc
CFLAGS := -O3 -march=native -lpthread
TARGET := bench

.PHONY: all clean run

all: $(TARGET)

$(TARGET): bench.c
	$(CC) $(CFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

perf: $(TARGET)
	perf stat -e dTLB-loads,dTLB-load-misses ./$(TARGET)

clean:
	rm -f $(TARGET)
