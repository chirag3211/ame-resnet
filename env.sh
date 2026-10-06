# source this in every new shell:  source env.sh
[ -f "$HOME/venv/bin/activate" ] && source "$HOME/venv/bin/activate"
export RISCV=$HOME/riscv
export PATH=$RISCV/bin:$HOME/buddy-mlir/llvm/build/bin:$HOME/buddy-mlir/build/bin:$PATH
export PYTHONPATH=$HOME/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:$HOME/buddy-mlir/build/python_packages
