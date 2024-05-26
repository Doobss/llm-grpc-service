use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(out_dir.join("service_v1_descriptor.bin"))
        .compile(&["protos/llm/service.proto"], &["protos"])?;
    tonic_build::compile_protos("protos/llm/service.proto").unwrap();
    Ok(())
}
