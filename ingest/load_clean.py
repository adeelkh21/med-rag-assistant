"""
Load and validate medical knowledge dataset from JSONL format.

This module:
- Reads JSONL file line-by-line
- Validates required fields (id, text, topic, source, source_type)
- Applies text quality checks
- Preserves metadata exactly as-is
- Returns validated document objects
"""

import orjson
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def validate_field(value: Any, field_name: str, field_type: type) -> bool:
    """
    Validate a single field.
    
    Args:
        value: Field value to validate
        field_name: Name of the field
        field_type: Expected type
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(value, field_type):
        return False
    
    if field_type == str and not value.strip():
        return False
    
    return True


def validate_text_quality(text: str) -> bool:
    """
    Apply text quality validation rules.
    
    Args:
        text: Text to validate
    
    Returns:
        True if text passes quality checks, False otherwise
    """
    # Must be a string
    if not isinstance(text, str):
        return False
    
    # Must not be empty or whitespace-only
    if not text.strip():
        return False
    
    # Must contain more than 40 words
    word_count = count_words(text)
    if word_count <= 40:
        return False
    
    return True


def validate_record(record: Dict[str, Any]) -> bool:
    """
    Validate a single record.
    
    Args:
        record: JSON object from JSONL
    
    Returns:
        True if record is valid, False otherwise
    """
    required_fields = {
        "id": str,
        "text": str,
        "topic": str,
        "source": str,
        "source_type": str
    }
    
    # Check all required fields exist and have correct type
    for field, field_type in required_fields.items():
        if field not in record:
            return False
        
        if not validate_field(record[field], field, field_type):
            return False
    
    # Apply text quality validation
    if not validate_text_quality(record["text"]):
        return False
    
    return True


def create_document(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a document object with preserved metadata.
    
    Args:
        record: Validated record from JSONL
    
    Returns:
        Document object with text and metadata
    """
    return {
        "text": record["text"],
        "metadata": {
            "id": record["id"],
            "topic": record["topic"],
            "source": record["source"],
            "source_type": record["source_type"]
        }
    }


def load_and_validate_dataset(jsonl_path: str) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Load and validate medical knowledge dataset from JSONL file.
    
    Reads file line-by-line to handle large files safely.
    Validates required fields and text quality.
    Preserves metadata exactly as-is.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        Tuple of (validated_documents, total_records, dropped_records)
    """
    path = Path(jsonl_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    validated_documents = []
    total_records = 0
    dropped_records = 0
    
    # Count total lines for progress bar
    total_lines = sum(1 for _ in open(jsonl_path, 'rb'))
    
    # Read and process line-by-line
    with open(jsonl_path, 'rb') as f:
        for line in tqdm(f, total=total_lines, desc="Loading dataset"):
            total_records += 1
            
            try:
                # Parse JSON using orjson (fast parsing)
                record = orjson.loads(line)
                
                # Validate record
                if validate_record(record):
                    document = create_document(record)
                    validated_documents.append(document)
                else:
                    dropped_records += 1
            
            except Exception as e:
                # Silently drop malformed records
                dropped_records += 1
    
    return validated_documents, total_records, dropped_records


def main():
    """Main entry point for dataset loading."""
    import sys
    
    # Default path (can be overridden via command line)
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        jsonl_path = Path(__file__).parent.parent / "data" / "medical_knowledge.jsonl"
    
    print(f"Loading dataset from: {jsonl_path}")
    
    documents, total, dropped = load_and_validate_dataset(str(jsonl_path))
    
    print(f"\n✓ Dataset Loaded Successfully")
    print(f"  Total records: {total}")
    print(f"  Valid documents: {len(documents)}")
    print(f"  Dropped records: {dropped}")
    
    if documents:
        print(f"\n  Sample document (first):")
        sample = documents[0]
        print(f"    Text length: {len(sample['text'])} chars")
        print(f"    Word count: {count_words(sample['text'])} words")
        print(f"    Metadata: {sample['metadata']}")
    
    return documents


if __name__ == "__main__":
    main()
