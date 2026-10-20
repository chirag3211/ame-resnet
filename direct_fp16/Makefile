# Direct (hand-written) fp16 AME kernels. Separate from the MLIR flow; nothing here touches ../Makefile.
#   make host-test | ops-host-test | test        host emulation (gcc only)
#   make spike-test | ops-spike-test             real run on the matrix Spike (needs env.sh)
#   make spike-test KC=64                        K-chunk length; each KC gets its own build dir, no `clean` needed
RISCV ?= $(HOME)/riscv
SPIKE ?= $(HOME)/riscv-stc/bin/spike
CC_RV ?= riscv64-unknown-elf-gcc
ISA   ?= rv64imafdcv_zfh_zicntr_matrix
KC    ?= 128
B     := build/kc$(KC)
HDRS  := ame16_hw.h ame16_kernels.h ame16_ops.h weights_inc.h
BIN   ?= $(HOME)/riscv-matrix-project/riscv-dnn/test/models/resnet50/bindata
IMG   ?= 0
REF   ?= reference/ref_softmax_fp16.bin
CF    := -O2 -DAME16_KC=$(KC) -I../include -I.
RVF   := -march=rv64gc_zfh -mabi=lp64d

test: host-test ops-host-test

host-test: $(B)/test_kernels_host
	./$<
ops-host-test: $(B)/test_ops_host
	./$<
$(B)/%_host: %.c $(HDRS)
	@mkdir -p $(B)
	gcc -Wall $(CF) $< -o $@ -lm

spike-test: $(B)/test_kernels.elf
	$(SPIKE) --isa=$(ISA) -m4096 $(RISCV)/riscv64-unknown-elf/bin/pk $<
ops-spike-test: $(B)/test_ops.elf
	$(SPIKE) --isa=$(ISA) -m4096 $(RISCV)/riscv64-unknown-elf/bin/pk $<
$(B)/%.elf: %.c $(HDRS)
	@mkdir -p $(B)
	$(CC_RV) $(CF) $(RVF) $< -o $@ -lm

# ---- full network (needs bindata: see gen_weights_inc.py). IMG = image index 0..68, KC as above.
$(B)/weights_inc.c: gen_weights_inc.py
	@mkdir -p $(B)
	python3 gen_weights_inc.py $(BIN) > $@
$(B)/net_host: resnet50_direct.c $(B)/weights_inc.c $(HDRS)
	gcc -O2 $(CF) resnet50_direct.c $(B)/weights_inc.c -o $@ -lm
$(B)/net.elf: resnet50_direct.c $(B)/weights_inc.c $(HDRS)
	$(CC_RV) $(CF) $(RVF) resnet50_direct.c $(B)/weights_inc.c -o $@ -lm
net-host: $(B)/net_host
	./$< $(IMG) | tee $(B)/net_host_img$(IMG).txt | grep -v "^LAYER\|^result"
	python3 check_result.py $(B)/net_host_img$(IMG).txt $(REF)
net-spike: $(B)/net.elf
	$(SPIKE) --isa=$(ISA) -m4096 $(RISCV)/riscv64-unknown-elf/bin/pk $< $(IMG) | tee $(B)/net_img$(IMG).txt | grep -v "^LAYER\|^result"
	python3 check_result.py $(B)/net_img$(IMG).txt $(REF) && python3 record_direct.py $(B)/net_img$(IMG).txt PASS \
	  || python3 record_direct.py $(B)/net_img$(IMG).txt FAIL

clean:
	rm -rf build
.PHONY: net-host net-spike test host-test ops-host-test spike-test ops-spike-test clean
