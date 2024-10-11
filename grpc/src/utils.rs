pub fn default_to_optional<T>(value: T) -> Option<T>
where
    T: Default + PartialEq,
{
    if value == T::default() {
        None
    } else {
        Some(value)
    }
}
