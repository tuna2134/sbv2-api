FROM rust AS builder
WORKDIR /work
COPY . .
RUN cargo build -r --bin sbv2_api -F cuda,cuda_tf32

FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
WORKDIR /work
COPY --from=builder /work/target/release/sbv2_api /work/main
COPY --from=builder /work/target/release/*.so /work
CMD ["/work/main"]