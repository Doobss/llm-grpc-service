pub mod llm;

use crate::Result;
use tonic_reflection::server::{ServerReflection, ServerReflectionServer};

pub mod v1 {
    tonic::include_proto!("llm.service");
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("llm_service_v1_descriptor");
}

pub fn spec_service() -> Result<ServerReflectionServer<impl ServerReflection>> {
    let spec = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(v1::FILE_DESCRIPTOR_SET)
        .build()?;
    Ok(spec)
}
