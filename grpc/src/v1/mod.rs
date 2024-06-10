mod messages;
pub mod pb;
pub mod services;
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("v1_service_descriptor");
