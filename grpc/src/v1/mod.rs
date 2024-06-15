mod messages;
pub mod pb;
pub mod services;
pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!("pb/v1_service_descriptor.bin");
