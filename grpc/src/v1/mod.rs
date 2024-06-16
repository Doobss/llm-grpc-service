mod messages;
mod pb;

pub mod llm {
    pub use super::pb::v1::llm::service::*;
}
pub mod prompt {
    pub use super::pb::v1::prompt::service::*;
}

pub mod services;
pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!("pb/service_descriptor.bin");
