# RAG Medical Pipeline - Phase 2

A Retrieval-Augmented Generation (RAG) system for medical knowledge retrieval and answer generation.

## Project Overview

This repository implements a RAG pipeline over trusted public medical guidance documents.
- **No model training required**
- **No re-chunking allowed** (chunks are already optimal)
- Focus: ingestion, embedding generation, retrieval, and generation

**Current Status**: ✅ STEPS 1-11 Complete (Full RAG Pipeline + Comprehensive Evaluation)

## Repository Structure

```
Phase2_RAG/
├── data/
│   └── medical_knowledge.jsonl          # Medical knowledge dataset (~43K chunks)
│
├── ingest/
│   ├── load_clean.py                    # ✅ Load & validate dataset
│   └── chunk_verify.py                  # ✅ Verification & statistics
│
├── embeddings/
│   ├── build_index.py                   # ✅ STEP 2: Generate embeddings
│   ├── embeddings.npy                   # ✅ Generated embeddings (43,207 x 1024)
│   ├── metadata.pkl                     # ✅ Metadata for all documents
│   └── config.pkl                       # ✅ Model configuration
│
├── retrieval/
│   ├── build_faiss_index.py             # ✅ STEP 3: Build FAISS index
│   ├── retriever.py                     # ✅ STEP 4: Semantic retriever
│   ├── index.faiss                      # ✅ FAISS index (43,207 vectors)
│   └── metadata_lookup.pkl              # ✅ Document metadata
│
├── generation/
│   ├── safety_filter.py                 # ✅ STEP 5: Pre-LLM safety gate
│   ├── prompts.py                       # ✅ STEP 6: Prompt templates
│   ├── llm_client.py                    # ✅ STEP 7: Groq API client
│   ├── validator.py                     # ✅ STEP 8: Response validator
│   └── answer_generator.py              # ✅ Main answer pipeline
│
├── evaluation/
│   ├── evaluation_dataset.json          # ✅ 50 test queries
│   ├── eval_retrieval.py                # ✅ STEP 10: Evaluation pipeline
│   ├── error_analysis.py                # ✅ STEP 11: Error diagnosis
│   ├── discover_chunks.py               # ✅ Helper for ground truth
│   ├── evaluation_results.json          # ✅ Generated results
│   └── evaluation_report.json           # ✅ Comprehensive report
│
├── app.py                               # Main entry point
├── requirements.txt                     # Python dependencies
├── validate_step5_8.py                  # ✅ Pipeline validation
├── test_pipeline_no_api.py              # ✅ Component tests
├── STEP5-8_README.md                    # ✅ Pipeline documentation
├── STEP5-8_COMPLETE.md                  # ✅ Completion summary
└── README.md                            # This file
```

## Setup Instructions

### Hardware Requirements

- **GPU Recommended**: NVIDIA GPU with CUDA support (e.g., RTX 4090)
- **CPU Alternative**: Embeddings can run on CPU (much slower)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~200MB for embeddings, ~1.5GB for model cache

### Python Environment

Requires **Python 3.10+** (tested with Python 3.12.7)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Required Dependencies

- `orjson>=3.9.0` - Fast JSON parsing
- `tqdm>=4.66.0` - Progress bars
- `pydantic>=2.0.0` - Data validation
- `numpy>=1.24.0` - Numerical computing
- `sentence-transformers>=2.2.0` - Text embeddings
- `faiss-cpu>=1.7.4` - Vector similarity search
- `groq>=0.4.0` - Groq API SDK
- `torch` - PyTorch for GPU acceleration (CUDA support)

## Usage

### Load & Validate Dataset (STEP 1 ✅)

```bash
python ingest/load_clean.py
```

**Output:**
```
✓ Dataset Loaded Successfully
  Total records: 43,363
  Valid documents: 43,207
  Dropped records: 156
```

### Verify Dataset (STEP 1.8 ✅)

```bash
python ingest/chunk_verify.py
```

**Output:**
```
✓ Document Verification Report
  Total documents: 43,207
  
  Word Count Statistics:
    Min:    41 words
    Max:    357 words
    Mean:   157.0 words
    Median: 137 words
  
  ✓ All documents exceed 40-word minimum
  ✓ All documents have complete metadata
  
  Unique topics: 9,297
  Unique sources: 10
```

### Generate Embeddings (STEP 2 ✅)

```bash
# Generate embeddings for all documents
python embeddings/build_index.py

# With custom batch size (for GPU)
python embeddings/build_index.py --batch-size 128

# Verify embeddings after generation
python embeddings/build_index.py --verify
```

**Output:**
```
======================================================================
STEP 2: Embedding Generation & Vector Preparation
======================================================================

[1/4] Loading validated documents...
✓ Loaded 43207 valid documents (dropped 156)

[2/4] Loading embedding model...
Loading embedding model: BAAI/bge-large-en-v1.5
Using device: CUDA
  GPU: NVIDIA GeForce RTX 4090
✓ Model loaded (embedding dimension: 1024)

[3/4] Generating embeddings...
Batches: 100%|████████████████████| 338/338 [03:16<00:00,  1.72it/s]
✓ Embeddings generated: shape (43207, 1024)
  ✓ All embeddings are normalized (L2 norm ≈ 1.0)

[4/4] Extracting metadata and saving...
✓ Embeddings saved to: embeddings/embeddings.npy (168.78 MB)
✓ Metadata saved to: embeddings/metadata.pkl
✓ Configuration saved to: embeddings/config.pkl

✓ STEP 2 Complete: Embeddings and metadata persisted
```

### Validate Complete Pipeline (Optional)

```bash
python validate_steps.py
```

**Validates:**
- ✅ STEP 1: Dataset loading and validation
- ✅ STEP 2: Embedding generation and quality
- ✅ Data consistency between steps

### Run Main Application

```bash
python app.py
```

## Implementation Status

### ✅ STEP 0: Repository & Environment Setup
- ✅ Directory structure created
- ✅ Python environment configured (3.12.7 with venv)
- ✅ Dependencies installed (all required packages)
- ✅ GPU acceleration configured (PyTorch with CUDA 12.1)

### ✅ STEP 1: Load & Validate Dataset

#### Features Implemented:
1. **Line-by-line JSONL loading** - Memory-efficient processing
2. **Fast JSON parsing** - Using `orjson` for performance
3. **Field validation** - All 5 required fields (id, text, topic, source, source_type)
4. **Text quality checks** - Must be >40 words, non-empty, string type
5. **Metadata preservation** - Exact preservation, no modifications
6. **Document structure**:
   ```python
   {
     "text": "Medical knowledge chunk text...",
     "metadata": {
       "id": "NCI_CANCER_TREAT_01",
       "topic": "Treatment of Cancer",
       "source": "National Cancer Institute (NCI, NIH)",
       "source_type": "cancer_research"
     }
   }
   ```

#### Validation Results:
- **Total records**: 43,363
- **Valid documents**: 43,207 (99.6% retention)
- **Dropped records**: 156 (0.4%)
- **Word count range**: 41-357 words (mean: 157, median: 137)
- **Unique topics**: 9,297
- **Unique sources**: 10

### ✅ STEP 2: Embedding Generation & Vector Preparation

#### Features Implemented:
1. **Embedding Model**: BAAI/bge-large-en-v1.5 (optimized for retrieval)
2. **Embedding Dimension**: 1024-dimensional vectors
3. **GPU Acceleration**: CUDA support with automatic device detection
4. **Normalization**: All embeddings L2-normalized for cosine similarity
5. **Batch Processing**: Configurable batch sizes (optimized for GPU)
6. **Persistence**: Embeddings + metadata saved to disk
7. **Query Instruction**: Documented for future retrieval steps

#### Generation Results:
- **Documents embedded**: 43,207
- **Embedding dimension**: 1024
- **Output size**: 168.78 MB
- **Processing time**: ~3-5 minutes (GPU) vs hours (CPU)
- **Normalization**: All vectors have L2 norm = 1.0
- **Device used**: CUDA (NVIDIA GeForce RTX 4090)

#### Files Generated:
- `embeddings/embeddings.npy` - Normalized embedding vectors (43,207 x 1024)
- `embeddings/metadata.pkl` - Complete metadata with text preservation
- `embeddings/config.pkl` - Model configuration and settings

#### Embedding Quality:
- ✅ All embeddings properly normalized (L2 norm ≈ 1.0)
- ✅ High embedding diversity (all dimensions have variance)
- ✅ Similarity sanity check passed (each doc most similar to itself)
- ✅ Metadata preserved exactly from STEP 1

#### Non-Goals (Explicitly Avoided):
- ❌ No FAISS index building yet (STEP 3)
- ❌ No retrieval implementation yet (STEP 3)
- ❌ No LLM calls
- ❌ No query processing yet

## Success Criteria

### ✅ STEP 1 Criteria (Complete)
- ✅ Repository structure exists in Phase2_RAG root
- ✅ Dataset loads without memory issues (line-by-line)
- ✅ Invalid records dropped (156 records, 99.6% retention)
- ✅ Metadata preserved exactly (no modifications)
- ✅ No chunking performed (chunks already optimal)

### ✅ STEP 2 Criteria (Complete)
- ✅ BAAI/bge-large-en-v1.5 model used
- ✅ Only document text embedded (not metadata)
- ✅ Embeddings normalized (L2 norm = 1.0)
- ✅ Metadata preserved exactly
- ✅ Embeddings persisted to disk
- ✅ No retrieval or LLM calls occurred

## Next Steps

### ✅ STEP 3: FAISS Index Construction (Complete)
- Built FAISS IndexFlatIP for exact inner product search
- 43,207 vectors indexed (1024 dimensions)
- Index persisted to disk (`retrieval/index.faiss`)
- Validation: All vectors retrievable

### ✅ STEP 4: Semantic Retriever (Complete)
- Implemented top-k document retrieval
- Query encoding with instruction prefix
- Configurable retrieval (default k=6)
- Deterministic results for evaluation

### ✅ STEP 5-8: Answer Generation Pipeline (Complete)
- **Safety Filter**: Blocks diagnosis/medication/treatment queries
- **Prompt Engineering**: Citation-grounded templates (locked)
- **Groq API**: LLaMA-3 70B integration (deterministic)
- **Response Validator**: Citation + disclaimer checks
- **Main Pipeline**: End-to-end orchestration

See [STEP5-8_README.md](STEP5-8_README.md) for full documentation.

### STEP 9: End-to-End Testing (Not Started)
- Create test dataset (50-100 queries)
- Run full pipeline on test set
- Collect success/failure metrics

### STEP 10: Evaluation Pipeline (Not Started)
- Implement faithfulness metrics
- Safety metric evaluation
- Retrieval quality assessment

## Key Design Decisions

1. **Line-by-line processing**: Memory efficiency for large datasets
2. **orjson parser**: Fastest JSON parser in Python (~150K records/sec)
3. **Strict validation**: Drop invalid records vs. attempting repair
4. **Metadata preservation**: No transformation ensures traceability
5. **Modular structure**: Clear separation for scalability
6. **BGE-large model**: Optimized for retrieval tasks (not just similarity)
7. **GPU acceleration**: 50-100x faster embedding generation vs CPU
8. **Normalized embeddings**: Required for efficient cosine similarity
9. **Batch processing**: Configurable batch sizes for hardware optimization

## Performance Notes

### STEP 1 (Dataset Loading)
- Loading time: ~0.3 seconds for 43,363 records
- Throughput: ~150,000 records/second
- Memory usage: Minimal (streaming, not loading full dataset)

### STEP 2 (Embedding Generation)
- **GPU (RTX 4090)**: ~3-5 minutes (batch size 128)
- **CPU**: ~2-4 hours (not recommended)
- Model download: 1.34 GB (one-time, cached)
- Output size: 168.78 MB (embeddings + metadata)
- Throughput: ~200-300 documents/second (GPU)

---

**Status**: STEPS 1-8 Complete ✅  
**Ready for**: STEP 9 (End-to-End Testing) & STEP 10 (Evaluation)

**Completion Summary:**
- ✅ 43,207 medical documents loaded and validated
- ✅ 43,207 embeddings generated with BAAI/bge-large-en-v1.5
- ✅ All embeddings normalized (L2 norm = 1.0)
- ✅ FAISS index built (IndexFlatIP, exact search)
- ✅ Semantic retriever implemented (top-k, deterministic)
- ✅ Safety filter blocks unsafe queries (diagnosis/medication/treatment)
- ✅ Citation-grounded generation with LLaMA-3 70B (via Groq)
- ✅ Response validation (citations + disclaimer)
- ✅ End-to-end pipeline ready for testing

**Quick Start (with Groq API key):**
```powershell
# Set API key
$env:GROQ_API_KEY = "gsk_your_key_here"

# Validate pipeline
python validate_step5_8.py

# Interactive demo
python generation/answer_generator.py
```
