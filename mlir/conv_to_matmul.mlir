// FIRST DRAFT (untested): rewrite conv_2d_nchw_fchw into img2col + matmul.
// Applied with:
//   mlir-opt in.mlir \
//     --transform-preload-library="transform-library-paths=mlir/conv_to_matmul.mlir" \
//     --transform-interpreter
// Op/pass spellings change between LLVM versions; if mlir-opt rejects this,
// paste the error and we will adapt it.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %img2col, %matmul = transform.structured.convert_conv2d_to_img2col %convs
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
