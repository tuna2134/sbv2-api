FROM ubuntu:latest
RUN apt update && apt install openssl libssl-dev curl pkg-config -y