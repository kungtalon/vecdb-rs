use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("failed to initialize new index: {0}")]
    InitializationError(String),
    #[error("failed to insert new data: {0}")]
    InsertionError(String),
    #[error("got unexpected error from index: {0}")]
    UnexpectedError(String),
}