use thiserror::Error;

#[derive(Debug, Error)]
#[error("{message}")]
pub struct InvalidParamsError {
    tool: &'static str,
    message: String,
    path: Option<String>,
}

impl InvalidParamsError {
    pub fn new(tool: &'static str, message: impl Into<String>) -> Self {
        Self {
            tool,
            message: message.into(),
            path: None,
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn tool(&self) -> &'static str {
        self.tool
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }
}
