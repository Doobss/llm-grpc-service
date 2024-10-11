ARG APP="server"


FROM rust:bookworm as chef
ARG APP
WORKDIR "/app"
RUN echo Building app=${APP} in dir /app
RUN apt-get update
RUN apt-get install pkg-config libssl-dev ca-certificates -y
RUN cargo install cargo-chef --locked


FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json


FROM chef as builder
ARG APP

COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin ${APP}
RUN rm src/*.rs




FROM debian:bookworm-slim as runtime
ARG APP
WORKDIR "/app"

RUN apt update
RUN apt-get install ca-certificates -y

RUN mkdir bin dist .config
COPY --from=builder "/app/.env" ./
COPY --from=builder "/app/target/release/${APP}" ./bin
COPY --from=builder "/app/dist" ./dist
COPY --from=builder "/app/.config" ./.config
RUN chmod +x /app/start.sh
CMD [ "sh", "-c", "start.sh", "${APP}" ]