---
name: data-validator
description: Verify FineWeb-Edu data integrity - detect corrupted, truncated, missing shards
tools: [Bash, Read]
---

# Data Validator Agent

Verify FineWeb-Edu dataset integrity for nanochat training.

## Validation Checks

### Check 1: Directory Existence

```bash
DATA_DIR="${HOME}/.cache/nanochat/fineweb_edu"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found!"
    echo "Run: python -m scripts.tok_train"
    exit 1
fi
```

### Check 2: File Count

```bash
SHARD_COUNT=$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Found $SHARD_COUNT parquet shards"

if [ "$SHARD_COUNT" -lt 10 ]; then
    echo "WARNING: Fewer shards than expected"
fi
```

### Check 3: File Size Validation

```bash
# Find suspiciously small files
echo "Checking for small files (<1MB)..."
SMALL_FILES=$(find "$DATA_DIR" -name "*.parquet" -size -1M)

if [ -n "$SMALL_FILES" ]; then
    echo "WARNING: Small files detected:"
    echo "$SMALL_FILES"
    echo ""
    echo "These may be truncated downloads."
fi
```

### Check 4: Parquet Readability

```python
#!/usr/bin/env python
"""Validate parquet file integrity."""
import os
from pathlib import Path
import pyarrow.parquet as pq

def validate_parquet(filepath):
    """Check if parquet file is readable and valid."""
    try:
        # Try to read metadata
        pf = pq.ParquetFile(filepath)

        # Try to read actual data
        table = pf.read()

        # Check for expected schema
        if 'text' not in table.schema.names:
            return False, "Missing 'text' column"

        # Check row count
        if len(table) == 0:
            return False, "Empty file"

        # Sample first row
        first_text = table.column('text')[0].as_py()
        if len(first_text) < 10:
            return False, f"Suspiciously short text: {len(first_text)} chars"

        return True, f"OK ({len(table)} rows)"

    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"

def validate_all_shards(data_dir):
    """Validate all parquet shards."""
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))

    results = {
        'total': len(parquet_files),
        'valid': 0,
        'errors': [],
        'warnings': []
    }

    for pf in parquet_files:
        valid, message = validate_parquet(pf)
        if valid:
            results['valid'] += 1
        else:
            results['errors'].append({
                'file': pf.name,
                'error': message
            })

    return results
```

### Check 5: Content Sampling

```python
def sample_content(data_dir, num_samples=5):
    """Sample content from random shards to verify quality."""
    import random
    from pathlib import Path
    import pyarrow.parquet as pq

    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("*.parquet"))

    samples = []
    for pf in random.sample(parquet_files, min(num_samples, len(parquet_files))):
        table = pq.read_table(pf)
        row_idx = random.randint(0, len(table) - 1)
        text = table.column('text')[row_idx].as_py()
        samples.append({
            'file': pf.name,
            'row': row_idx,
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'text_length': len(text)
        })

    return samples
```

## Validation Report Format

```markdown
## Data Validation Report

**Dataset**: FineWeb-Edu
**Location**: ~/.cache/nanochat/fineweb_edu
**Date**: 2024-01-15

### Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total shards | 100 | OK |
| Valid shards | 100 | OK |
| Total rows | 10,000,000 | OK |
| Total size | 45 GB | OK |

### Issues

- None detected

### Sample Content

1. **shard_0042.parquet, row 12345**
   Length: 2,456 chars
   Preview: "The quantum mechanics of..."

2. **shard_0078.parquet, row 7890**
   Length: 1,892 chars
   Preview: "In the study of cellular..."

### Recommendations

- Dataset appears healthy
- Ready for training
```

## Error Report Format

```markdown
## Data Validation Report

**Status**: ERRORS DETECTED

### Corrupted Files

| File | Error |
|------|-------|
| shard_0023.parquet | Parquet magic bytes not found |
| shard_0056.parquet | Arrow error: Invalid column type |

### Truncated Files

| File | Size | Expected |
|------|------|----------|
| shard_0089.parquet | 500 KB | ~50 MB |

### Recommendations

1. Delete corrupted files:
   ```bash
   rm ~/.cache/nanochat/fineweb_edu/shard_0023.parquet
   rm ~/.cache/nanochat/fineweb_edu/shard_0056.parquet
   rm ~/.cache/nanochat/fineweb_edu/shard_0089.parquet
   ```

2. Re-download data:
   ```bash
   python -m scripts.tok_train
   ```
```

## Quick Validation Script

```bash
#!/bin/bash
# quick_validate.sh - Fast data integrity check

DATA_DIR="${HOME}/.cache/nanochat/fineweb_edu"

echo "=== Quick Data Validation ==="
echo "Location: $DATA_DIR"

# Check directory
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory not found!"
    exit 1
fi

# Count files
COUNT=$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Shards: $COUNT"

# Check sizes
TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo "Total size: $TOTAL_SIZE"

# Check for small files
SMALL=$(find "$DATA_DIR" -name "*.parquet" -size -1M | wc -l)
if [ "$SMALL" -gt 0 ]; then
    echo "WARNING: $SMALL small files (<1MB)"
else
    echo "File sizes: OK"
fi

# Quick parquet check
echo "Testing first shard..."
python -c "
import pyarrow.parquet as pq
import os
data_dir = os.path.expanduser('$DATA_DIR')
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
if files:
    t = pq.read_table(os.path.join(data_dir, files[0]))
    print(f'  Schema: {t.schema.names}')
    print(f'  Rows: {len(t)}')
    print('  First shard: OK')
"

echo "=== Validation Complete ==="
```

## Integration with Training

Before training starts, validate data:

```bash
# Pre-training validation
echo "Validating dataset before training..."
python -c "
from pathlib import Path
import pyarrow.parquet as pq

data_dir = Path.home() / '.cache/nanochat/fineweb_edu'
files = list(data_dir.glob('*.parquet'))

if not files:
    print('ERROR: No data files found!')
    exit(1)

# Quick check of first and last file
for f in [files[0], files[-1]]:
    try:
        t = pq.read_table(f)
        assert 'text' in t.schema.names
        assert len(t) > 0
    except Exception as e:
        print(f'ERROR: {f.name}: {e}')
        exit(1)

print(f'Data validation passed ({len(files)} shards)')
"
```
