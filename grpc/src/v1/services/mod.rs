pub mod llm;
pub mod prompt;

use crate::v1;
use crate::Result;
use tonic_reflection::server::{ServerReflection, ServerReflectionServer};

pub fn spec_service() -> Result<ServerReflectionServer<impl ServerReflection>> {
    tracing::info!("Adding spec service");
    let spec = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(v1::FILE_DESCRIPTOR_SET)
        .build()?;
    Ok(spec)
}
