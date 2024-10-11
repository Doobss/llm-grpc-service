use super::DeviceResult;
use std::{fmt::Debug, sync::Arc};

#[derive(Clone)]
pub struct Device {
    inner: Arc<cudarc::driver::CudaDevice>,
    memory_size: usize,
}

impl Device {
    pub fn from_ordinal(ordinal: i32) -> DeviceResult<Self> {
        let inner = cudarc::driver::CudaDevice::new(ordinal as usize)?;
        let memory_size = unsafe { cudarc::driver::result::device::total_mem(ordinal) }?;
        Ok(Self { inner, memory_size })
    }

    pub fn memory(&self) -> usize {
        self.memory_size
    }

    pub fn name(&self) -> DeviceResult<String> {
        Ok(self.inner.name()?)
    }

    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    pub fn candle_device(&self) -> DeviceResult<candle_core::Device> {
        Ok(candle_core::Device::new_cuda(self.ordinal())?)
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("ordinal", &self.ordinal())
            .field("memory_size", &self.memory_size)
            .finish()
    }
}
