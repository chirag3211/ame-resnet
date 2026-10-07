# ---------------------------------------------------------------------------
#  Phase A  : kernel API tests          (make test-native | make test-spike)
#  Phase B  : one linalg.matmul via MLIR (make mlir-matmul-spike)
#  Phase C  : ResNet18                   (see README, scripts in mlir/ + python/)
# ---------------------------------------------------------------------------
RISCV      ?= $(HOME)/riscv
CROSS      ?= riscv64-unknown-elf-
ARCH       ?= rv64gc
ABI        ?= lp64d
SPIKE      ?= spike
SPIKE_ISA  ?= rv64gc
SPIKE_MEM  ?= 2048
PK         ?= $(RISCV)/riscv64-unknown-elf/bin/pk
USE_AME    ?= 0
B          := build

CFLAGS  := -O2 -Wall -Wextra -Iinclude
LIB     := src/ref_matmul.c src/ame_dispatch.c src/memref_shims.c
ifeq ($(USE_AME),1)
CFLAGS  += -DUSE_AME
ifeq ($(AME_COUNT_STEPS),1)
CFLAGS  += -DAME_COUNT_STEPS
endif
LIB     += src/ame_hw.c
endif

RVCC    := $(CROSS)gcc -march=$(ARCH) -mabi=$(ABI)

.PHONY: all test-native test-spike mlir-matmul-spike resnet-spike clean

all: test-native

# ----- Phase A (host) -----
$(B)/test_matmul_native: tests/test_matmul.c $(LIB)
	gcc $(CFLAGS) $^ -o $@
$(B)/test_shim_native: tests/test_shim_native.c $(LIB)
	gcc $(CFLAGS) $^ -o $@
test-native: $(B)/test_matmul_native $(B)/test_shim_native
	./$(B)/test_matmul_native
	./$(B)/test_shim_native

# ----- Phase A (Spike) -----
$(B)/test_matmul.elf: tests/test_matmul.c $(LIB)
	$(RVCC) $(CFLAGS) $^ -o $@
test-spike: $(B)/test_matmul.elf
	$(SPIKE) --isa=$(SPIKE_ISA) -m$(SPIKE_MEM) $(PK) $<

# ----- Phase B: MLIR matmul -> call into ame_matmul -> Spike -----
MLIR_OPT   ?= mlir-opt
MLIR_TR    ?= mlir-translate
LLC        ?= llc

$(B)/matmul_calls.mlir: mlir/matmul.mlir mlir/matmul_to_call.py
	python3 mlir/matmul_to_call.py $< $@
$(B)/matmul_llvm.mlir: $(B)/matmul_calls.mlir
	$(MLIR_OPT) $< \
	  --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
	  --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm \
	  --convert-func-to-llvm --reconcile-unrealized-casts -o $@
$(B)/generated_shims.c: $(B)/matmul_llvm.mlir mlir/gen_shims.py
	python3 mlir/gen_shims.py $< $@
$(B)/matmul.ll: $(B)/matmul_llvm.mlir
	$(MLIR_TR) --mlir-to-llvmir $< -o $@
$(B)/matmul.o: $(B)/matmul.ll
	$(LLC) $< -mtriple=riscv64-unknown-elf -mattr=+m,+a,+f,+d,+c \
	  -target-abi=$(ABI) -filetype=obj -o $@
$(B)/matmul_mlir.elf: tests/test_mlir_matmul.c $(B)/generated_shims.c $(B)/matmul.o $(LIB)
	$(RVCC) $(CFLAGS) -I. $^ -o $@
mlir-matmul-spike: $(B)/matmul_mlir.elf
	$(SPIKE) --isa=$(SPIKE_ISA) -m$(SPIKE_MEM) $(PK) $<

# ----- Phase C: ResNet18 (needs build/resnet18.mlir from python/export_resnet18.py) -----
SIZE ?= 224
$(B)/resnet18_llvm.mlir: $(B)/resnet18.mlir mlir/lower_resnet.sh mlir/matmul_to_call.py
	bash mlir/lower_resnet.sh $< $@
$(B)/resnet18_shims.c: $(B)/resnet18_llvm.mlir mlir/gen_shims.py
	python3 mlir/gen_shims.py $< $@
$(B)/resnet18.ll: $(B)/resnet18_llvm.mlir
	$(MLIR_TR) --mlir-to-llvmir $< -o $@
# LLVM_OPT=O2|O3 runs LLVM's IR optimizer (opt) before llc; default: none (as measured so far).
# Like USE_AME, it is not tracked: rm -f $(B)/*.o when changing it.
LLVM_OPT     ?=
LLVM_OPT_BIN ?= opt
$(B)/resnet18.o: $(B)/resnet18.ll
	@if [ -n "$(LLVM_OPT)" ]; then \
	  $(LLVM_OPT_BIN) -passes='default<$(LLVM_OPT)>' -mtriple=riscv64-unknown-elf -mattr=+m,+a,+f,+d,+c $< -S -o $(B)/resnet18_opt.ll; \
	  cp $(B)/resnet18_opt.ll $(B)/resnet18_cg.ll; \
	 else cp $< $(B)/resnet18_cg.ll; fi
	$(LLC) $(B)/resnet18_cg.ll -mtriple=riscv64-unknown-elf -mattr=+m,+a,+f,+d,+c \
	  -target-abi=$(ABI) -filetype=obj -o $@
# embeds $(B)/input.bin -> symbol _binary_input_bin_start (objcopy runs inside $(B) so the symbol is path-independent)
$(B)/input.o: $(B)/input.bin
	cd $(B) && $(CROSS)objcopy -I binary -O elf64-littleriscv -B riscv input.bin input.o
DRIVER  ?= src/main_resnet.c
COMPARE ?= python/compare.py
$(B)/resnet18.elf: $(DRIVER) src/memref_copy.c $(B)/resnet18_shims.c $(B)/resnet18.o $(B)/input.o $(LIB)
	$(RVCC) $(CFLAGS) -DINPUT_H=$(SIZE) -DINPUT_W=$(SIZE) -I. $^ -o $@ -lm
resnet-spike: $(B)/resnet18.elf
	bash -o pipefail -c '$(SPIKE) --isa=$(SPIKE_ISA) -m$(SPIKE_MEM) $(PK) $< | tee $(B)/spike_out.txt' \
	  || { rc=$$?; echo "SPIKE EXITED rc=$$rc (137=killed/OOM, 1/2=pk trap or program exit)"; exit $$rc; }
	python3 $(COMPARE) $(B)/spike_out.txt $(B)/golden_logits.bin

clean:
	find $(B) -mindepth 1 ! -name "resnet18.mlir" ! -name "input.bin" ! -name "golden_logits.bin" ! -name "meta.json" -delete
