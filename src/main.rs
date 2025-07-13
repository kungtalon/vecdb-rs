mod filter;
mod index;
mod merror;
mod persistence;
mod scalar;
mod vecdb;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::post,
    Router,
};
use axum_extra::extract::WithRejection;
use axum_macros::debug_handler;
use futures::lock::Mutex;
use merror::ApiError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::{net::SocketAddr, sync::Arc};
use tracing::{event, span, Level};
use tracing_subscriber::fmt as tracing_fmt;

use vecdb::{DatabaseParams, DocMap, VectorDatabase, VectorInsertArgs, VectorSearchArgs};

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseParams,
    pub file_path: String,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub search_url_suffix: String,
    pub upsert_url_suffix: String,
    pub port: u16,
    pub log_level: String,
}

#[derive(Debug, Serialize)]
struct VectorSearchResponse {
    results: Vec<DocMap>, // pretend these are doc IDs or similar
}

#[derive(Debug, Serialize)]
struct VectorUpsertResponse {
    message: String,
}

#[debug_handler]
async fn handle_vector_search(
    State(vdb): State<Arc<Mutex<VectorDatabase>>>,
    WithRejection(Json(payload), _): WithRejection<Json<VectorSearchArgs>, ApiError>,
) -> (StatusCode, Json<VectorSearchResponse>) {
    let span = span!(Level::TRACE, "handle_vector_search");
    let _enter = span.enter();

    event!(
        Level::INFO,
        "Received search request with payload: {:?}",
        payload
    );

    let search_args = VectorSearchArgs {
        query: payload.query,
        filter_inputs: payload.filter_inputs,
        k: payload.k,
        hnsw_params: payload.hnsw_params,
    };

    let results = {
        let mut vdb_guard = vdb.lock().await;

        vdb_guard.query(search_args).await
    };

    match results {
        Ok(results) => {
            event!(Level::INFO, "Search successful");
            (StatusCode::OK, Json(VectorSearchResponse { results }))
        }
        Err(e) => {
            event!(Level::ERROR, "Error during vector search: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(VectorSearchResponse { results: vec![] }),
            )
        }
    }
}

#[debug_handler]
async fn handle_vector_upsert(
    State(vdb): State<Arc<Mutex<VectorDatabase>>>,
    WithRejection(Json(payload), _): WithRejection<Json<VectorInsertArgs>, ApiError>,
) -> (StatusCode, Json<VectorUpsertResponse>) {
    let span = span!(Level::TRACE, "handle_vector_upsert");
    let _enter = span.enter();

    event!(
        Level::INFO,
        "Received upsert request with payload: {:?}",
        payload
    );

    let results = {
        let mut vdb_guard = vdb.lock().await;

        vdb_guard.upsert(payload).await
    };

    match results {
        Ok(_) => {
            event!(Level::INFO, "Upsert successful");
            let response = VectorUpsertResponse {
                message: "Upsert successful".to_string(),
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            event!(Level::ERROR, "Error during vector upsert: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(VectorUpsertResponse {
                    message: format!("Error during vector upsert: {e}"),
                }),
            )
        }
    }
}

fn parse_settings() -> Result<AppConfig, config::ConfigError> {
    let setting = config::Config::builder()
        .add_source(config::File::with_name("config.toml"))
        .build()?;

    let dec_setting: AppConfig = setting.try_deserialize()?;

    Ok(dec_setting)
}

#[tokio::main]
async fn main() {
    let app_config = parse_settings().unwrap();

    let subscriber = tracing_fmt()
        .with_max_level(Level::from_str(app_config.server.log_level.as_str()).unwrap())
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();

    let vdb: VectorDatabase =
        VectorDatabase::new(app_config.file_path, app_config.database).unwrap();
    let vdb_state = Arc::new(Mutex::new(vdb));

    let app = Router::new()
        .route(
            &app_config.server.search_url_suffix,
            post(handle_vector_search),
        )
        .route(
            &app_config.server.upsert_url_suffix,
            post(handle_vector_upsert),
        )
        .with_state(vdb_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], app_config.server.port));
    println!("Server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");

    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
