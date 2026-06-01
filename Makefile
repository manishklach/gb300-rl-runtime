CC      := gcc
NVCC    := nvcc
CFLAGS  := -O3 -Wall -Wextra -march=native -Iinclude
NVFLAGS := -O3 -Iinclude --extended-lambda \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=sm_90a
LDFLAGS := -lpthread -lnuma -lcudart

SRCDIR  := src
CUDIR   := cu
BLDDIR  := build

MAIN_SRC := $(SRCDIR)/main.c
C_SRCS  := $(filter-out $(MAIN_SRC), $(wildcard $(SRCDIR)/*.c))
CU_SRCS := $(wildcard $(CUDIR)/*.cu)
C_OBJS  := $(patsubst $(SRCDIR)/%.c, $(BLDDIR)/%.o, $(C_SRCS))
MAIN_OBJ := $(BLDDIR)/main.o
CU_OBJS := $(patsubst $(CUDIR)/%.cu, $(BLDDIR)/%.o, $(CU_SRCS))

TESTDIR := test
TEST_SRCS  := $(wildcard $(TESTDIR)/*.cu) $(wildcard $(TESTDIR)/*.c)
TEST_TARGET := $(BLDDIR)/test_bench
SMOKE_SRC := $(TESTDIR)/test_smoke.c
SMOKE_TARGET := $(BLDDIR)/test_smoke
HW_TEST_SRC := $(TESTDIR)/test_hw_ring.c
HW_TEST_TARGET := $(BLDDIR)/test_hw_ring

BENCHDIR := bench
BENCH_SRC := $(BENCHDIR)/bench_pipeline.cu
BENCH_TARGET := $(BLDDIR)/bench_pipeline
BENCH_TRACE_SRC := $(BENCHDIR)/bench_trace_pipeline.cu
BENCH_TRACE_TARGET := $(BLDDIR)/bench_trace_pipeline
BENCH_COW_SRC := $(BENCHDIR)/bench_cow_prefix.cu
BENCH_COW_TARGET := $(BLDDIR)/bench_cow_prefix
BENCH_TAX_SRC := $(BENCHDIR)/bench_control_tax.cu
BENCH_TAX_TARGET := $(BLDDIR)/bench_control_tax
BENCH_GPU_SCHED_SRC := $(BENCHDIR)/bench_gpu_scheduler.cu
BENCH_GPU_SCHED_TARGET := $(BLDDIR)/bench_gpu_scheduler
BENCH_DECODE_SRC := $(BENCHDIR)/bench_decode_microkernel.cu
BENCH_DECODE_TARGET := $(BLDDIR)/bench_decode_microkernel
BENCH_KV_LAYOUT_SRC := $(BENCHDIR)/bench_kv_layout.cu
BENCH_KV_LAYOUT_TARGET := $(BLDDIR)/bench_kv_layout
BENCH_PREFETCH_SRC := $(BENCHDIR)/bench_prefetch.cu
BENCH_PREFETCH_TARGET := $(BLDDIR)/bench_prefetch
BENCH_HW_FASTPATH_SRC := $(BENCHDIR)/bench_hw_fastpath.c
BENCH_HW_FASTPATH_TARGET := $(BLDDIR)/bench_hw_fastpath

BENCH_TARGETS := $(BENCH_TARGET) $(BENCH_TRACE_TARGET) $(BENCH_COW_TARGET) $(BENCH_TAX_TARGET) $(BENCH_GPU_SCHED_TARGET) $(BENCH_DECODE_TARGET) $(BENCH_KV_LAYOUT_TARGET) $(BENCH_PREFETCH_TARGET) $(BENCH_HW_FASTPATH_TARGET)
CUDA_CHECK_DIR := $(BLDDIR)/cuda_check
CUDA_CHECK_SRCS := $(MAIN_SRC) $(CU_SRCS) $(TESTDIR)/test_bench.cu \
                   $(BENCHDIR)/bench_pipeline.cu $(BENCHDIR)/bench_trace_pipeline.cu \
                   $(BENCHDIR)/bench_cow_prefix.cu $(BENCHDIR)/bench_gpu_scheduler.cu \
                   $(BENCHDIR)/bench_decode_microkernel.cu $(BENCHDIR)/bench_kv_layout.cu \
                   $(BENCHDIR)/bench_prefetch.cu
CUDA_PTX_SRCS := $(CU_SRCS) $(BENCHDIR)/bench_decode_microkernel.cu $(BENCHDIR)/bench_prefetch.cu

.PHONY: all clean smoke test test-hw-ring test-all rtl-test rtl-clean bench bench-pipeline bench-trace bench-cow bench-tax bench-gpu-scheduler bench-decode bench-kv-layout bench-prefetch bench-hw-fastpath bench-all ci-build ci-run cuda-compile-check cuda-ptx-check

all: $(BLDDIR)/libruntime.a $(TEST_TARGET) $(HW_TEST_TARGET) $(BENCH_TARGETS)

$(BLDDIR):
	mkdir -p $(BLDDIR)

$(CUDA_CHECK_DIR):
	mkdir -p $(CUDA_CHECK_DIR)

$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC) $(BLDDIR)
	$(NVCC) $(NVFLAGS) -x cu -c $< -o $@

$(BLDDIR)/%.o: $(CUDIR)/%.cu $(BLDDIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BLDDIR)/test_bench.o: $(TESTDIR)/test_bench.cu $(BLDDIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BLDDIR)/libruntime.a: $(C_OBJS) $(MAIN_OBJ) $(CU_OBJS)
	ar rcs $@ $^

$(TEST_TARGET): $(BLDDIR)/test_bench.o $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_TARGET): $(BENCH_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_TRACE_TARGET): $(BENCH_TRACE_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_COW_TARGET): $(BENCH_COW_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_TAX_TARGET): $(BENCH_TAX_SRC)
	$(CC) $(CFLAGS) -x c $< -lpthread -o $@

$(BENCH_GPU_SCHED_TARGET): $(BENCH_GPU_SCHED_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_DECODE_TARGET): $(BENCH_DECODE_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_KV_LAYOUT_TARGET): $(BENCH_KV_LAYOUT_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_PREFETCH_TARGET): $(BENCH_PREFETCH_SRC) $(BLDDIR)/libruntime.a
	$(NVCC) $(NVFLAGS) $< -L$(BLDDIR) -lruntime $(LDFLAGS) -o $@

$(BENCH_HW_FASTPATH_TARGET): $(BENCH_HW_FASTPATH_SRC) $(SRCDIR)/hw_ring.c $(SRCDIR)/infer_submit.c $(SRCDIR)/hw_worker_sim.c $(SRCDIR)/mmio.c $(BLDDIR)
	$(CC) $(CFLAGS) $(BENCH_HW_FASTPATH_SRC) $(SRCDIR)/hw_ring.c $(SRCDIR)/infer_submit.c $(SRCDIR)/hw_worker_sim.c $(SRCDIR)/mmio.c -lpthread -o $@

$(SMOKE_TARGET): $(SMOKE_SRC) $(SRCDIR)/ring.c $(SRCDIR)/pipeline.c $(SRCDIR)/rollout.c $(SRCDIR)/decode_batch.c $(BLDDIR)
	$(CC) $(CFLAGS) $(SMOKE_SRC) $(SRCDIR)/ring.c $(SRCDIR)/pipeline.c $(SRCDIR)/rollout.c $(SRCDIR)/decode_batch.c -o $@

$(HW_TEST_TARGET): $(HW_TEST_SRC) $(SRCDIR)/hw_ring.c $(SRCDIR)/infer_submit.c $(SRCDIR)/hw_worker_sim.c $(SRCDIR)/mmio.c $(BLDDIR)
	$(CC) $(CFLAGS) $(HW_TEST_SRC) $(SRCDIR)/hw_ring.c $(SRCDIR)/infer_submit.c $(SRCDIR)/hw_worker_sim.c $(SRCDIR)/mmio.c -lpthread -o $@

smoke: $(SMOKE_TARGET)
	./$(SMOKE_TARGET)

test-hw-ring: $(HW_TEST_TARGET)
	./$(HW_TEST_TARGET)

test: smoke test-hw-ring $(TEST_TARGET)
	./$(TEST_TARGET)

test-all: test
	$(MAKE) rtl-test

rtl-test:
	@if command -v iverilog >/dev/null 2>&1; then \
		mkdir -p $(BLDDIR); \
		iverilog -g2012 -o $(BLDDIR)/rtl_tb \
			rtl/desc_pkg.sv \
			rtl/mmio_regs.sv \
			rtl/desc_ring.sv \
			rtl/completion_ring.sv \
			rtl/rollout_worker_fsm.sv \
			rtl/rl_runtime_top.sv \
			rtl/tb_rl_runtime_top.sv && \
		vvp $(BLDDIR)/rtl_tb; \
	else \
		echo "Install iverilog or use Verilator to run RTL tests."; \
	fi

rtl-clean:
	rm -f $(BLDDIR)/rtl_tb

bench: $(TEST_TARGET)
	./$(TEST_TARGET) --bench 1000000

bench-pipeline: $(BENCH_TARGET)
	./$(BENCH_TARGET)

bench-trace: $(BENCH_TRACE_TARGET)
	./$(BENCH_TRACE_TARGET)

bench-cow: $(BENCH_COW_TARGET)
	./$(BENCH_COW_TARGET)

bench-tax: $(BENCH_TAX_TARGET)
	./$(BENCH_TAX_TARGET)

bench-gpu-scheduler: $(BENCH_GPU_SCHED_TARGET)
	./$(BENCH_GPU_SCHED_TARGET)

bench-decode: $(BENCH_DECODE_TARGET)
	./$(BENCH_DECODE_TARGET)

bench-kv-layout: $(BENCH_KV_LAYOUT_TARGET)
	./$(BENCH_KV_LAYOUT_TARGET)

bench-prefetch: $(BENCH_PREFETCH_TARGET)
	./$(BENCH_PREFETCH_TARGET)

bench-hw-fastpath: $(BENCH_HW_FASTPATH_TARGET)
	./$(BENCH_HW_FASTPATH_TARGET) $(ARGS)

bench-all: bench-hw-fastpath bench bench-pipeline bench-trace bench-cow bench-tax bench-gpu-scheduler bench-decode bench-kv-layout bench-prefetch

LABS := 01_false_sharing 02_spsc_ring 03_hugepage_tlb 04_syscall_vs_poll 05_doorbell_mock 06_memory_ordering
LABS_CPU_SAFE := 01_false_sharing 02_spsc_ring 04_syscall_vs_poll 05_doorbell_mock 06_memory_ordering

.PHONY: labs lab-run lab-run-safe lab-clean

labs:
	@for lab in $(LABS); do \
		echo "=== Building lab/$$lab ==="; \
		$(MAKE) -C lab/$$lab || exit $$?; \
	done

lab-run:
	@for lab in $(LABS); do \
		echo "=== Running lab/$$lab ==="; \
		$(MAKE) -C lab/$$lab run || exit $$?; \
		echo; \
	done

lab-run-safe:
	@echo "=== Running lab/01_false_sharing ==="
	@$(MAKE) -C lab/01_false_sharing run ARGS="1000000"
	@echo
	@echo "=== Running lab/02_spsc_ring ==="
	@$(MAKE) -C lab/02_spsc_ring run ARGS="1000000"
	@echo
	@echo "=== Running lab/04_syscall_vs_poll ==="
	@$(MAKE) -C lab/04_syscall_vs_poll run ARGS="10000"
	@echo
	@echo "=== Running lab/05_doorbell_mock ==="
	@$(MAKE) -C lab/05_doorbell_mock run ARGS="200000"
	@echo
	@echo "=== Running lab/06_memory_ordering ==="
	@$(MAKE) -C lab/06_memory_ordering run ARGS="200000"
	@echo

ci-build: $(SMOKE_TARGET) $(HW_TEST_TARGET) $(BENCH_TAX_TARGET) $(BENCH_HW_FASTPATH_TARGET) labs

ci-run: smoke
	./$(HW_TEST_TARGET)
	./$(BENCH_TAX_TARGET) 100000
	./$(BENCH_HW_FASTPATH_TARGET) 10000
	$(MAKE) lab-run-safe

cuda-compile-check: $(CUDA_CHECK_DIR)
	@for src in $(CUDA_CHECK_SRCS); do \
		base=$$(basename $$src); \
		out="$(CUDA_CHECK_DIR)/$${base%.*}.o"; \
		echo "=== nvcc compile-check $$src ==="; \
		$(NVCC) $(NVFLAGS) -c $$src -o $$out || exit $$?; \
	done

cuda-ptx-check: $(CUDA_CHECK_DIR)
	@for src in $(CUDA_PTX_SRCS); do \
		base=$$(basename $$src); \
		out="$(CUDA_CHECK_DIR)/$${base%.*}.ptx"; \
		echo "=== nvcc ptx-check $$src ==="; \
		$(NVCC) $(NVFLAGS) -ptx $$src -o $$out || exit $$?; \
	done

lab-clean:
	@for lab in $(LABS); do \
		$(MAKE) -C lab/$$lab clean || exit $$?; \
	done

clean:
	rm -rf $(BLDDIR)
	$(MAKE) lab-clean
