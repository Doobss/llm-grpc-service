use super::GenerationRequest;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GenerationBatch {
    requests: HashMap<String, GenerationRequest>,
}

impl GenerationBatch {
    pub fn from_requests(requests: Vec<GenerationRequest>) -> Self {
        let mut mapped_requests = HashMap::with_capacity(requests.len());
        for request in requests {
            let id = request.id.clone();
            mapped_requests.insert(id, request);
        }
        Self {
            requests: mapped_requests,
        }
    }
}

impl GenerationBatch {
    pub fn len(&self) -> usize {
        self.requests.keys().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
