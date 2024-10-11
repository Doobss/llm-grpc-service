use super::{Device, DeviceResult};

#[derive(Debug)]
pub struct DeviceMap {
    pub devices: Vec<Device>,
}

impl DeviceMap {
    pub fn new() -> DeviceResult<Self> {
        let number_of_devices = cudarc::driver::result::device::get_count()?;
        let devices: DeviceResult<Vec<Device>> =
            (0..number_of_devices).map(Device::from_ordinal).collect();
        let devices = devices?;
        Ok(Self { devices })
    }

    pub fn total_memory(&self) -> usize {
        self.devices
            .iter()
            .fold(0_usize, |mem, device| mem + device.memory())
    }

    pub fn number_of_devices(&self) -> usize {
        self.devices.len()
    }
}
