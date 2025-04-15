use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::time::Instant;

/// High-performance Rust extensions for LlamaDB.
///
/// This module provides Rust implementations of performance-critical
/// operations for LlamaDB, including vector similarity calculations,
/// text processing, and data manipulation.
#[pymodule]
fn llamadb_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_text, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_vector_ops, m)?)?;
    m.add_class::<VectorIndex>()?;
    Ok(())
}

/// Calculate cosine similarity between vectors.
///
/// Args:
///     a: First vector or matrix of vectors
///     b: Second vector or matrix of vectors
///
/// Returns:
///     Cosine similarity score(s)
#[pyfunction]
fn cosine_similarity<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let a_view = a.as_array();
    let b_view = b.as_array();
    
    // Calculate dot products and norms
    let a_norm = (a_view.dot(&a_view)).sqrt();
    
    // Use Rayon for parallel processing
    let mut similarities = Array1::zeros(b_view.shape()[0]);
    similarities
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut sim)| {
            let b_vec = b_view.row(i);
            let b_norm = (b_vec.dot(&b_vec)).sqrt();
            let dot_product = a_view.dot(&b_vec);
            
            // Avoid division by zero
            if a_norm > 0.0 && b_norm > 0.0 {
                sim[0] = dot_product / (a_norm * b_norm);
            } else {
                sim[0] = 0.0;
            }
        });
    
    Ok(similarities.into_pyarray(py))
}

/// Calculate Euclidean distance between vectors.
///
/// Args:
///     a: First vector or matrix of vectors
///     b: Second vector or matrix of vectors
///
/// Returns:
///     Euclidean distance(s)
#[pyfunction]
fn euclidean_distance<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let a_view = a.as_array();
    let b_view = b.as_array();
    
    // Use Rayon for parallel processing
    let mut distances = Array1::zeros(b_view.shape()[0]);
    distances
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut dist)| {
            let b_vec = b_view.row(i);
            let squared_diff: f32 = a_view
                .iter()
                .zip(b_vec.iter())
                .map(|(a_val, b_val)| (a_val - b_val).powi(2))
                .sum();
            
            dist[0] = squared_diff.sqrt();
        });
    
    Ok(distances.into_pyarray(py))
}

/// Normalize a batch of vectors to unit length.
///
/// Args:
///     vectors: Matrix of vectors to normalize
///
/// Returns:
///     Normalized vectors
#[pyfunction]
fn batch_normalize<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let vectors_view = vectors.as_array();
    let shape = vectors_view.shape();
    
    // Create output array
    let mut normalized = Array2::zeros((shape[0], shape[1]));
    
    // Use Rayon for parallel processing
    normalized
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut row)| {
            let vec = vectors_view.row(i);
            let norm: f32 = vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
            
            // Avoid division by zero
            if norm > 0.0 {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = vec[j] / norm;
                }
            }
        });
    
    Ok(normalized.into_pyarray(py))
}

/// Tokenize text into words.
///
/// Args:
///     text: Input text to tokenize
///
/// Returns:
///     List of tokens
#[pyfunction]
fn tokenize_text(py: Python, text: &str) -> PyResult<Vec<String>> {
    // Simple tokenization by splitting on whitespace and punctuation
    let tokens: Vec<String> = text
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    
    Ok(tokens)
}

/// Benchmark vector operations.
///
/// Args:
///     size: Size of vectors to benchmark
///     iterations: Number of iterations to run
///
/// Returns:
///     Dictionary with benchmark results
#[pyfunction]
fn benchmark_vector_ops(py: Python, size: usize, iterations: usize) -> PyResult<PyObject> {
    // Create random vectors
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
    let b: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..size).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    // Convert to ndarray
    let a_array = Array1::from_vec(a);
    let b_array = Array2::from_shape_vec(
        (b.len(), size),
        b.into_iter().flatten().collect(),
    )?;
    
    // Benchmark cosine similarity
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = cosine_similarity_impl(&a_array, &b_array);
    }
    let cosine_time = start.elapsed().as_secs_f64() / iterations as f64;
    
    // Benchmark Euclidean distance
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = euclidean_distance_impl(&a_array, &b_array);
    }
    let euclidean_time = start.elapsed().as_secs_f64() / iterations as f64;
    
    // Benchmark normalization
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = batch_normalize_impl(&b_array);
    }
    let normalize_time = start.elapsed().as_secs_f64() / iterations as f64;
    
    // Create result dictionary
    let result = PyDict::new(py);
    result.set_item("cosine_time", cosine_time)?;
    result.set_item("euclidean_time", euclidean_time)?;
    result.set_item("normalize_time", normalize_time)?;
    result.set_item("vector_size", size)?;
    result.set_item("batch_size", b_array.shape()[0])?;
    
    Ok(result.into())
}

// Internal implementation functions

fn cosine_similarity_impl(a: &ArrayView1<f32>, b: &ArrayView2<f32>) -> Array1<f32> {
    let a_norm = (a.dot(a)).sqrt();
    
    let mut similarities = Array1::zeros(b.shape()[0]);
    similarities
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut sim)| {
            let b_vec = b.row(i);
            let b_norm = (b_vec.dot(&b_vec)).sqrt();
            let dot_product = a.dot(&b_vec);
            
            // Avoid division by zero
            if a_norm > 0.0 && b_norm > 0.0 {
                sim[0] = dot_product / (a_norm * b_norm);
            } else {
                sim[0] = 0.0;
            }
        });
    
    similarities
}

fn euclidean_distance_impl(a: &ArrayView1<f32>, b: &ArrayView2<f32>) -> Array1<f32> {
    let mut distances = Array1::zeros(b.shape()[0]);
    distances
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut dist)| {
            let b_vec = b.row(i);
            let squared_diff: f32 = a
                .iter()
                .zip(b_vec.iter())
                .map(|(a_val, b_val)| (a_val - b_val).powi(2))
                .sum();
            
            dist[0] = squared_diff.sqrt();
        });
    
    distances
}

fn batch_normalize_impl(vectors: &ArrayView2<f32>) -> Array2<f32> {
    let shape = vectors.shape();
    
    // Create output array
    let mut normalized = Array2::zeros((shape[0], shape[1]));
    
    // Use Rayon for parallel processing
    normalized
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut row)| {
            let vec = vectors.row(i);
            let norm: f32 = vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
            
            // Avoid division by zero
            if norm > 0.0 {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = vec[j] / norm;
                }
            }
        });
    
    normalized
}

/// High-performance vector index implemented in Rust.
///
/// This class provides a simple but efficient vector index for similarity search,
/// implemented in Rust for maximum performance.
#[pyclass]
struct VectorIndex {
    dimension: usize,
    metric: String,
    vectors: Option<Array2<f32>>,
    metadata: Vec<PyObject>,
}

#[pymethods]
impl VectorIndex {
    /// Create a new vector index.
    ///
    /// Args:
    ///     dimension: Dimensionality of vectors to be indexed
    ///     metric: Similarity metric, one of "cosine" or "euclidean"
    #[new]
    fn new(py: Python, dimension: usize, metric: String) -> PyResult<Self> {
        if metric != "cosine" && metric != "euclidean" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported metric: {}. Use 'cosine' or 'euclidean'.", metric),
            ));
        }
        
        Ok(VectorIndex {
            dimension,
            metric,
            vectors: None,
            metadata: Vec::new(),
        })
    }
    
    /// Add vectors to the index.
    ///
    /// Args:
    ///     vectors: Numpy array of shape (n, dimension) with vectors to add
    ///     metadata: Optional list of metadata dictionaries, one per vector
    fn add(&mut self, py: Python, vectors: PyReadonlyArray2<f32>, metadata: Option<Vec<PyObject>>) -> PyResult<()> {
        let vectors_view = vectors.as_array();
        
        // Check dimensions
        if vectors_view.shape()[1] != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected vectors of dimension {}, got {}", self.dimension, vectors_view.shape()[1]),
            ));
        }
        
        // Convert to owned array
        let vectors_array = vectors_view.to_owned();
        
        // Normalize vectors for cosine similarity
        let normalized_vectors = if self.metric == "cosine" {
            batch_normalize_impl(&vectors_array.view())
        } else {
            vectors_array.clone()
        };
        
        // Add vectors to index
        match &mut self.vectors {
            Some(existing_vectors) => {
                let mut new_vectors = Array2::zeros((
                    existing_vectors.shape()[0] + normalized_vectors.shape()[0],
                    self.dimension,
                ));
                
                // Copy existing vectors
                for (i, row) in existing_vectors.rows().into_iter().enumerate() {
                    new_vectors.row_mut(i).assign(&row);
                }
                
                // Copy new vectors
                for (i, row) in normalized_vectors.rows().into_iter().enumerate() {
                    new_vectors.row_mut(existing_vectors.shape()[0] + i).assign(&row);
                }
                
                *existing_vectors = new_vectors;
            }
            None => {
                self.vectors = Some(normalized_vectors);
            }
        }
        
        // Add metadata
        match metadata {
            Some(meta) => {
                if meta.len() != vectors_view.shape()[0] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!(
                            "Length of metadata ({}) must match number of vectors ({})",
                            meta.len(),
                            vectors_view.shape()[0]
                        ),
                    ));
                }
                self.metadata.extend(meta);
            }
            None => {
                // Create empty dictionaries as metadata
                for _ in 0..vectors_view.shape()[0] {
                    self.metadata.push(PyDict::new(py).into());
                }
            }
        }
        
        Ok(())
    }
    
    /// Search for the k most similar vectors to the query.
    ///
    /// Args:
    ///     query: Query vector as numpy array of shape (dimension,)
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     List of dictionaries with results, including id, score, and metadata
    fn search(&self, py: Python, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<Vec<PyObject>> {
        let query_view = query.as_array();
        
        // Check dimensions
        if query_view.shape()[0] != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Expected query of shape ({}), got {:?}",
                    self.dimension,
                    query_view.shape()
                ),
            ));
        }
        
        // Get vectors
        let vectors = match &self.vectors {
            Some(v) => v,
            None => {
                return Ok(Vec::new());
            }
        };
        
        // Calculate similarity scores
        let scores = if self.metric == "cosine" {
            // Normalize query for cosine similarity
            let query_norm = (query_view.dot(&query_view)).sqrt();
            let normalized_query = if query_norm > 0.0 {
                query_view.map(|&x| x / query_norm)
            } else {
                query_view.to_owned()
            };
            
            // Calculate cosine similarity
            cosine_similarity_impl(&normalized_query.view(), &vectors.view())
        } else {
            // Calculate Euclidean distance
            let distances = euclidean_distance_impl(&query_view, &vectors.view());
            
            // Convert to negative distances (higher is better)
            distances.map(|&x| -x)
        };
        
        // Find top k indices
        let mut indices_with_scores: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by score (descending)
        indices_with_scores.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take top k
        let top_k = indices_with_scores.into_iter().take(k);
        
        // Create result objects
        let mut results = Vec::new();
        for (idx, score) in top_k {
            let result = PyDict::new(py);
            result.set_item("id", idx)?;
            result.set_item("score", score)?;
            result.set_item("metadata", &self.metadata[idx])?;
            
            // Add vector
            let vector = vectors.row(idx).to_owned();
            result.set_item("vector", vector.into_pyarray(py))?;
            
            results.push(result.into());
        }
        
        Ok(results)
    }
    
    /// Return the number of vectors in the index.
    fn __len__(&self) -> usize {
        match &self.vectors {
            Some(v) => v.shape()[0],
            None => 0,
        }
    }
}
