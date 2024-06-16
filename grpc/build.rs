use std::{fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v1_out_dir = PathBuf::from("src/v1/pb");
    let v1_proto_files = [
        "protos/v1/llm/service.proto",
        "protos/v1/prompt/service.proto",
    ];

    fs::create_dir_all(&v1_out_dir).unwrap_or(());

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(v1_out_dir.join("service_descriptor.bin"))
        .out_dir(v1_out_dir)
        .include_file("mod.rs")
        .compile(&v1_proto_files, &["protos"])?;
    // recompile protobufs only if any of the proto files changes.
    for file in v1_proto_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    Ok(())
}
