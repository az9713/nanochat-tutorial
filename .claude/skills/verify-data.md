---
name: verify-data
description: Validate FineWeb-Edu dataset integrity - detect corrupted, truncated, or missing shards
---

# Data Integrity Validator

Verify FineWeb-Edu dataset integrity to prevent training issues.

## Quick Validation

Run this comprehensive check:

```bash
#!/bin/bash
# verify_data.sh - Quick data integrity check

DATA_DIR="${HOME}/.cache/nanochat/fineweb_edu"

echo "=== FineWeb-Edu Data Validation ==="
echo "Location: $DATA_DIR"
echo

# Check if directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found!"
    echo "Run: python -m scripts.tok_train to download data"
    exit 1
fi

# Count shards
SHARD_COUNT=$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Shards found: $SHARD_COUNT"

if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No parquet files found!"
    exit 1
fi

# Check file sizes
echo
echo "=== File Sizes ==="
echo "Checking for suspiciously small files..."
SMALL_FILES=$(find "$DATA_DIR" -name "*.parquet" -size -1M)
if [ -n "$SMALL_FILES" ]; then
    echo "WARNING: Found small files (< 1MB):"
    echo "$SMALL_FILES"
else
    echo "OK: All files are reasonable size"
fi

# Total size
TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo
echo "Total dataset size: $TOTAL_SIZE"

echo
echo "=== Validation Complete ==="
```

## Deep Validation (Python)

For thorough validation of parquet files:

```python
#!/usr/bin/env python
"""Deep validation of FineWeb-Edu dataset."""
import os
from pathlib import Path
import pyarrow.parquet as pq

def validate_dataset(data_dir=None):
    if data_dir is None:
        data_dir = Path.home() / ".cache/nanochat/fineweb_edu"
    else:
        data_dir = Path(data_dir)

    print(f"Validating: {data_dir}\n")

    if not data_dir.exists():
        print("ERROR: Data directory not found!")
        return False

    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files\n")

    errors = []
    warnings = []
    total_rows = 0

    for i, pf in enumerate(sorted(parquet_files)):
        try:
            # Try to read the file
            table = pq.read_table(pf)
            rows = len(table)
            total_rows += rows

            # Check for reasonable row count
            if rows < 100:
                warnings.append(f"{pf.name}: Only {rows} rows (suspiciously low)")

            # Check schema
            if 'text' not in table.schema.names:
                errors.append(f"{pf.name}: Missing 'text' column")

            # Sample check
            if rows > 0:
                sample = table.to_pandas().iloc[0]['text']
                if len(sample) < 10:
                    warnings.append(f"{pf.name}: First row text very short ({len(sample)} chars)")

            # Progress
            if (i + 1) % 10 == 0:
                print(f"Validated {i + 1}/{len(parquet_files)} files...")

        except Exception as e:
            errors.append(f"{pf.name}: {type(e).__name__}: {e}")

    print(f"\n=== Results ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\n=== ERRORS (must fix) ===")
        for e in errors:
            print(f"  ✗ {e}")

    if warnings:
        print("\n=== WARNINGS (review) ===")
        for w in warnings:
            print(f"  ! {w}")

    if not errors and not warnings:
        print("\n✓ All files validated successfully!")
        return True

    return len(errors) == 0

if __name__ == "__main__":
    validate_dataset()
```

## Common Issues

### Issue 1: Truncated Downloads

**Symptom:** File size is much smaller than expected.

**Cause:** Interrupted download, network timeout.

**Solution:**
```bash
# Remove corrupted shard and re-download
rm ~/.cache/nanochat/fineweb_edu/shard_XXXX.parquet

# Re-run data download
python -m scripts.tok_train
```

### Issue 2: Missing Shards

**Symptom:** Training fails looking for shard that doesn't exist.

**Diagnostic:**
```bash
# List available shards
ls ~/.cache/nanochat/fineweb_edu/*.parquet | sort -V

# Check for gaps in numbering
```

**Solution:**
```bash
# Re-download specific shard (if download script supports it)
# Or re-download all data
rm -rf ~/.cache/nanochat/fineweb_edu
python -m scripts.tok_train
```

### Issue 3: Parquet Read Errors

**Symptom:** `ArrowInvalid: Parquet magic bytes not found`

**Cause:** Completely corrupted file.

**Solution:**
```bash
# Identify and remove corrupted file
rm ~/.cache/nanochat/fineweb_edu/CORRUPTED_FILE.parquet

# Re-download
python -m scripts.tok_train
```

### Issue 4: Schema Mismatch

**Symptom:** Training expects column that doesn't exist.

**Diagnostic:**
```python
import pyarrow.parquet as pq

# Check schema of a file
table = pq.read_table('shard.parquet')
print(table.schema)
```

**Solution:**
- Verify you have correct dataset version
- Re-download if schema changed

## Expected Dataset Stats

For FineWeb-Edu (approximate):

| Metric | Expected Value |
|--------|----------------|
| Total shards | ~100 |
| Rows per shard | ~100,000 |
| File size | ~50-200MB each |
| Total size | ~10-50GB |
| Text column | String, non-empty |

## Quick Commands

```bash
# Check data location
echo "Data: ~/.cache/nanochat/fineweb_edu"
ls -la ~/.cache/nanochat/fineweb_edu/ | head -20

# Count files
ls ~/.cache/nanochat/fineweb_edu/*.parquet | wc -l

# Total size
du -sh ~/.cache/nanochat/fineweb_edu

# Check specific file
python -c "
import pyarrow.parquet as pq
t = pq.read_table('~/.cache/nanochat/fineweb_edu/shard_0000.parquet')
print(f'Rows: {len(t)}')
print(f'Schema: {t.schema}')
print(f'Sample: {t[0][0].as_py()[:100]}...')
"
```

## Re-download Commands

If validation fails and you need fresh data:

```bash
# Complete re-download
rm -rf ~/.cache/nanochat/fineweb_edu
python -m scripts.tok_train

# Or use a different cache location
export NANOCHAT_BASE_DIR=/path/to/new/cache
python -m scripts.tok_train
```

## Tokenizer Validation

Also verify the tokenizer:

```bash
# Check tokenizer files
ls -la ~/.cache/nanochat/tokenizer/

# Test tokenizer
python -c "
from nanochat.tokenizer import Tokenizer
tok = Tokenizer()
encoded = tok.encode('Hello, world!')
decoded = tok.decode(encoded)
print(f'Encoded: {encoded}')
print(f'Decoded: {decoded}')
assert decoded == 'Hello, world!', 'Tokenizer roundtrip failed!'
print('Tokenizer OK!')
"
```
