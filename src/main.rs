mod filter;
mod index;
mod merror;
mod scalar;
mod vecdb;

use axum::{
    extract::{rejection::JsonRejection, Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    BoxError, Router,
};
use axum_extra::extract::WithRejection;
use merror::ApiError;
use serde::Serialize;
use std::net::SocketAddr;
use tower::ServiceBuilder;
use tracing::{event, span, Level};
use tracing_subscriber::fmt as tracing_fmt;

use vecdb::{DocMap, IndexParams, VectorDatabase, VectorInsertArgs, VectorSearchArgs};

#[derive(Debug, Serialize)]
struct VectorSearchResponse {
    results: Vec<DocMap>, // pretend these are doc IDs or similar
}

#[derive(Debug, Serialize)]
struct VectorUpsertResponse {
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    message: String,
}

async fn handle_vector_search(
    State(mut vdb): State<VectorDatabase>,
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
    let results = vdb.query(search_args).await;

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

async fn handle_vector_upsert(
    State(mut vdb): State<VectorDatabase>,
    WithRejection(Json(payload), _): WithRejection<Json<VectorInsertArgs>, ApiError>,
) -> (StatusCode, Json<VectorUpsertResponse>) {
    let span = span!(Level::TRACE, "handle_vector_search");
    let _enter = span.enter();

    event!(
        Level::INFO,
        "Received upsert request with payload: {:?}",
        payload
    );

    let results = vdb.upsert(payload).await;

    match results {
        Ok(_) => {
            event!(Level::INFO, "Upsert successful");
            let response = VectorUpsertResponse {
                message: "Upsert successful".to_string(),
            };
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            event!(Level::ERROR, "Error during vector upsert: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(VectorUpsertResponse {
                    message: format!("Error during vector upsert: {}", e),
                }),
            )
        }
    }
}

#[tokio::main]
async fn main() {
    let subscriber = tracing_fmt().with_max_level(Level::TRACE).finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();

    let vdb: VectorDatabase = VectorDatabase::new(
        "./testdata",
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

    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
