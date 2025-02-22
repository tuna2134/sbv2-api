FROM rust AS builder
WORKDIR /work
COPY . .
RUN cargo build -r --bin sbv2_api -F cuda,cuda_tf32
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
WORKDIR /work
COPY --from=builder /work/target/release/sbv2_api /work/main
COPY --from=builder /work/target/release/*.so /work
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work
CMD ["/work/main"]