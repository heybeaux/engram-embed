# engram-embed: Local embedding server (Rust/Candle)
# Note: Metal acceleration is macOS-only. Docker builds use CPU.
FROM rust:1.77-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
# Create dummy src for dependency caching
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

COPY src ./src
COPY examples ./examples
COPY tests ./tests
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/engram-embed .

EXPOSE 8080
CMD ["./engram-embed"]
