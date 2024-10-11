use super::{ChatMessage, ChatTemplateInputs, TemplateResult};
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;

#[derive(Clone, Debug)]
pub struct ChatTemplate {
    template: Template<'static, 'static>,
    raw_template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

impl ChatTemplate {
    pub fn new(template: String, bos_token: Option<String>, eos_token: Option<String>) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        let raw_template = template.clone();
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);

        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        Self {
            template,
            bos_token,
            eos_token,
            raw_template,
        }
    }

    pub fn apply(&self, messages: Vec<ChatMessage>) -> TemplateResult<String> {
        Ok(self.template.render(ChatTemplateInputs {
            messages,
            bos_token: self.bos_token.as_deref(),
            eos_token: self.eos_token.as_deref(),
            add_generation_prompt: true,
        })?)
    }

    pub fn get_template(&self) -> String {
        self.raw_template.clone()
    }
}
