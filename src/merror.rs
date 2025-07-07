use axum::{
    extract::{rejection::JsonRejection, Json},
    response::IntoResponse,
};
use serde_json::json;
use std::fmt::Display;
use thiserror::Error;
use tracing::{event, Level};

#[derive(Error, Debug)]
#[allow(unused, clippy::enum_variant_names)]
pub enum DBError {
    #[error("failed to create database: {0}")]
    CreateError(String),
    #[error("failed to close database: {0}")]
    CloseError(String),
    #[error("failed to get data from database: {0}")]
    GetError(String),
    #[error("failed to put data into database: {0}")]
    PutError(String),
    #[error("failed to sync data in database: {0}")]
    SyncError(String),
    #[error("failed to delete data from database: {0}")]
    DeleteDataError(String),
}

#[derive(Error, Debug)]
#[allow(unused, clippy::enum_variant_names)]
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

#[derive(Debug, thiserror::Error)]
pub struct FileError(pub String);

impl Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub struct DataError(pub String);

impl Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    // The `#[from]` attribute generates `From<JsonRejection> for ApiError`
    // implementation. See `thiserror` docs for more information
    #[error(transparent)]
    JsonExtractorRejection(#[from] JsonRejection),
}

// We implement `IntoResponse` so ApiError can be used as a response
impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        event!(Level::INFO, "request rejected: {}", self);

        let (status, message) = match self {
            ApiError::JsonExtractorRejection(json_rejection) => {
                (json_rejection.status(), json_rejection.body_text())
            }
        };

        let payload = json!({
            "message": message,
        });

        (status, Json(payload)).into_response()
    }
}
