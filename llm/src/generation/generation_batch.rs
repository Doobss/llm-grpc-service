use super::GenerationRequest;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct GenerationBatch {
    pub requests: IndexMap<String, GenerationRequest>,
}

impl GenerationBatch {
    pub fn from_requests(requests: Vec<GenerationRequest>) -> Self {
        let mut mapped_requests = IndexMap::with_capacity(requests.len());
        for request in requests.into_iter() {
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

    pub fn get_requests(&self) -> Vec<&GenerationRequest> {
        self.requests.values().collect()
    }

    pub fn get_prompts(&self) -> Vec<String> {
        let mut prompts: Vec<String> = Vec::with_capacity(self.len());
        for request in self.requests.values() {
            prompts.push(request.content.clone());
        }
        prompts
    }
}
