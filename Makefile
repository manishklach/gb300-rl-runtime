CC      := gcc
NVCC    := nvcc
CFLAGS  := -O3 -Wall -Wextra -march=native -Iinclude
NVFLAGS := -O3 -arch=sm_90a -Iinclude --extended-lambda
LDFLAGS := -lpthread -lnuma -lcudart

SRCDIR  := src
CUDIR   := cu
BLDDIR  := build

C_SRCS  := $(wildcard $(SRCDIR)/*.c)
CU_SRCS := $(wildcard $(CUDIR)/*.cu)
C_OBJS  := $(patsubst $(SRCDIR)/%.c, $(BLDDIR)/%.o, $(C_SRCS))
CU_OBJS := $(patsubst $(CUDIR)/%.cu, $(BLDDIR)/%.o, $(CU_SRCS))

TESTDIR := test
TEST_SRCS  := $(wildcard $(TESTDIR)/*.cu) $(wildcard $(TESTDIR)/*.c)
TEST_TARGET := $(BLDDIR)/test_bench

BENCHDIR := bench
BENCH_SRC := $(BENCHDIR)/bench_pipeline.cu
BENCH_TARGET := $(BLDDIR)/bench_pipeline
BENCH_TRACE_SRC := $(BENCHDIR)/bench_trace_pipeline.cu
BENCH_TRACE_TARGET := $(BLDDIR)/bench_trace_pipeline

.PHONY: all clean test bench bench-pipeline bench-trace bench-all

all: $(BLDDIR)/libruntime.a $(TEST_TARGET) $(BENCH_TARGET) $(BENCH_TRACE_TARGET)

$(BLDDIR):
	mkdir -p $(BLDDIR)

$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BLDDIR)/%.o: $(CUDIR)/%.cu $(BLDDIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BLDDIR)/test_bench.o: $(TESTDIR)/test_bench.cu $(BLDDIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BLDDIR)/libruntime.a: $(C_OBJS) $(CU_OBJS)
	ar rcs $@ $^

$(TEST_TARGET): $(BLDDIR)/test_bench.o $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_TARGET): $(BENCH_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_TRACE_TARGET): $(BENCH_TRACE_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

bench: $(TEST_TARGET)
	./$(TEST_TARGET) --bench --tokens 1000000

bench-pipeline: $(BENCH_TARGET)
	./$(BENCH_TARGET)

bench-trace: $(BENCH_TRACE_TARGET)
	./$(BENCH_TRACE_TARGET)

bench-all: bench bench-pipeline bench-trace

LABS := 01_false_sharing 02_spsc_ring 03_hugepage_tlb 04_syscall_vs_poll 05_doorbell_mock

.PHONY: labs lab-clean

labs:
	@for lab in $(LABS); do \
		echo "=== Building lab/$$lab ==="; \
		$(MAKE) -C lab/$$lab; \
	done

lab-run:
	@for lab in $(LABS); do \
		echo "=== Running lab/$$lab ==="; \
		$(MAKE) -C lab/$$lab run; \
		echo; \
	done

lab-clean:
	@for lab in $(LABS); do \
		$(MAKE) -C lab/$$lab clean; \
	done

clean:
	rm -rf $(BLDDIR)
	$(MAKE) lab-clean
