// use std::{
//     fs::{create_dir_all, File},
//     path::Path,
//     sync::Arc,
// };
use tracing_subscriber::{filter, prelude::*};

// // construct a subscriber that prints formatted traces to stdout
// let subscriber = tracing_subscriber::FmtSubscriber::new();
// // use that subscriber to process traces emitted after this point
// tracing::subscriber::set_global_default(subscriber).unwrap();

pub fn init() {
    let stdout_log = tracing_subscriber::fmt::layer().pretty();
    // let log_directory = Path::new(".logs/server");
    // match create_dir_all(log_directory) {
    //     Ok(_) => (),
    //     Err(_) => panic!(
    //         "{}",
    //         format!("Cannot create server log directory @ {log_directory:#?}")
    //     ),
    // };
    // // A layer that logs events to a file.
    // let file = match File::create(".logs/server/debug.log") {
    //     Ok(file) => file,
    //     Err(error) => panic!("Error: {:?}", error),
    // };

    // let debug_log = tracing_subscriber::fmt::layer().with_writer(Arc::new(file));

    // A layer that collects metrics using specific events.
    // let metrics_layer = /* ... */ filter::LevelFilter::INFO;

    tracing_subscriber::registry()
        .with(stdout_log.with_filter(filter::LevelFilter::DEBUG))
        // .with(
        //     // Add a filter to the metrics label that *only* enables
        //     // events whose targets start with `metrics`.
        //     metrics_layer.with_filter(filter::filter_fn(|metadata| {
        //         metadata.target().starts_with("metrics")
        //     })),
        // )
        .init();
}
