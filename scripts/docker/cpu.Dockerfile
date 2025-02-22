FROM rust AS builder
WORKDIR /work
COPY . .
RUN cargo build -r --bin sbv2_api
FROM gcr.io/distroless/cc-debian12
WORKDIR /work
COPY --from=builder /work/target/release/sbv2_api /work/main
COPY --from=builder /work/target/release/*.so /work
CMD ["/work/main"]