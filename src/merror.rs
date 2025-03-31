use thiserror::Error;

#[derive(Error, Debug)]
pub enum DBError {
    #[error("failed to create database: {0}")]
    CreateError(String),
    #[error("failed to close database: {0}")]
    CloseError(String),
    #[error("failed to get data from database: {0}")]
    GetError(String),
    #[error("failed to put data into database: {0}")]
    PutError(String),
    #[error("failed to delete data from database: {0}")]
    DeleteDataError(String),
}

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("failed to initialize new index: {0}")]
    InitializationError(String),
    #[error("failed to insert new data: {0}")]
    InsertionError(String),
    #[error("failed to do index query: {0}")]
    QueryError(String),
    #[error("got unexpected error from index: {0}")]
    UnexpectedError(String),
}
