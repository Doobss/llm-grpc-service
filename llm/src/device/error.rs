pub type DeviceResult<T> = core::result::Result<T, DeviceError>;

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error(transparent)]
    CudaDriverError(#[from] cudarc::driver::DriverError),
}
