mod filter;
mod index;
mod merror;
mod scalar;
mod util;
mod vecdb;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::post,
    Router,
};
use serde::Serialize;
use std::net::SocketAddr;
use tracing::{event, span, Level};

use vecdb::{DocMap, IndexParams, VectorDatabase, VectorInsertArgs, VectorSearchArgs};

#[derive(Debug, Serialize)]
struct VectorSearchResponse {
    results: Vec<DocMap>, // pretend these are doc IDs or similar
}

#[derive(Debug, Serialize)]
struct VectorUpsertResponse {
    message: String,
}

async fn handle_vector_search(
    State(mut vdb): State<VectorDatabase>,
    Json(payload): Json<VectorSearchArgs>,
) -> (StatusCode, Json<VectorSearchResponse>) {
    let span = span!(Level::TRACE, "handle_vector_search", ?payload);
    let _enter = span.enter();

    let search_args = VectorSearchArgs {
        query: payload.query,
        filter_inputs: payload.filter_inputs,
        k: payload.k,
        hnsw_params: payload.hnsw_params,
    };
    let results = vdb.query(search_args).await.unwrap_or_else(|e| {
        event!(Level::ERROR, "Error during vector search: {}", e);
        vec![]
    });

    (StatusCode::OK, Json(VectorSearchResponse { results }))
}

async fn handle_vector_upsert(
    State(mut vdb): State<VectorDatabase>,
    Json(payload): Json<VectorInsertArgs>,
) -> (StatusCode, Json<VectorUpsertResponse>) {
    let span = span!(Level::TRACE, "handle_vector_search", ?payload);
    let _enter = span.enter();

    let results = vdb.upsert(payload).await;

    match results {
        Ok(_) => {
            event!(Level::INFO, "Upsert successful");
            let response = VectorUpsertResponse {
                message: "Upsert successful".to_string(),
            };
            return (StatusCode::OK, Json(response));
        }
        Err(e) => {
            event!(Level::ERROR, "Error during vector upsert: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(VectorUpsertResponse {
                    message: format!("Error during vector upsert: {}", e),
                }),
            );
        }
    }
}

#[tokio::main]
async fn main() {
    let vdb: VectorDatabase = VectorDatabase::new(
        "./test",
        IndexParams {
            dim: 128,
            index_type: index::IndexType::Flat,
            metric_type: index::MetricType::L2,
            hnsw_params: None,
        },
    )
    .unwrap();

    let app = Router::new()
        .route("/search", post(handle_vector_search))
        .route("/upsert", post(handle_vector_upsert))
        .with_state(vdb.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], 7000));
    println!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");

    axum::serve(listener, app).await.unwrap();
}
