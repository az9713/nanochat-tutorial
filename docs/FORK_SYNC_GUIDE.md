# Fork Sync Guide: Sync Fork with Upstream While Preserving Custom Features

## Situation
- **Upstream**: `karpathy/nanochat` (has latest updates from Karpathy)
- **Your fork**: `az9713/nanochat-with-claude` (has Claude Code integration + custom features)
- **Local**: `C:\Users\simon\Downloads\nanochat-master` (has new documentation)
- **Goal**: Get Karpathy's latest changes WITHOUT losing your custom features

## Your Custom Features (to preserve)
From your 20 merged PRs:
1. Interactive Tokenizer Playground
2. Training Progress Dashboard
3. Checkpoint Browser & Comparator
4. Dataset Inspector
5. Model Size & Cost Calculator
6. Generation Parameter Explorer
7. Training Resume Helper
8. Attention Visualizer
9. Learning Rate Finder
10. Conversation Template Builder
+ Bug fixes and documentation

## Recommended Approach

### Step 1: Clone Your Fork (fresh start)
```bash
cd C:\Users\simon\Downloads
git clone https://github.com/az9713/nanochat-with-claude.git nanochat-fork
cd nanochat-fork
```

### Step 2: Add Karpathy's Repo as Upstream
```bash
git remote add upstream https://github.com/karpathy/nanochat.git
git fetch upstream
```

### Step 3: Check for Divergence
```bash
# See how many commits your fork is ahead/behind upstream
git rev-list --left-right --count main...upstream/main
```

### Step 4: Merge Upstream into Your Fork
```bash
# This preserves your commits and adds Karpathy's new commits
git checkout main
git merge upstream/main
```

**If there are conflicts:**
- Git will show which files conflict
- Open conflicting files and resolve manually
- Common conflicts: files both you and Karpathy modified
- After resolving: `git add <file>` then `git commit`

### Step 5: Push Merged Changes
```bash
git push origin main
```

### Step 6: Add New Documentation
```bash
# Copy documentation from local directory
cp ../nanochat-master/CLAUDE.md .
mkdir -p docs
cp ../nanochat-master/docs/* docs/

# Commit
git add CLAUDE.md docs/
git commit -m "Add comprehensive documentation for developers and users

- CLAUDE.md: Guidance for Claude Code
- docs/QUICK_START.md: 12 hands-on use cases
- docs/USER_GUIDE.md: Complete user guide
- docs/DEVELOPER_GUIDE.md: Development guide
- docs/ARCHITECTURE.md: Technical architecture
- docs/REMOTE_GPU_SETUP.md: SSH tunneling for cloud GPUs
- docs/README.md: Documentation index

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push origin main
```

## Git Remote Layout
```
origin   -> https://github.com/az9713/nanochat-with-claude.git   (YOUR fork - push here)
upstream -> https://github.com/karpathy/nanochat.git (Karpathy's - pull from here)
```

## Future Syncing
Whenever Karpathy updates his repo:
```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## Verification
1. Run `git log --oneline -20` to see both your commits and Karpathy's
2. Check that your custom features still work (run tests if available)
3. Verify new documentation is present in `docs/` folder
4. Compare key files with Karpathy's repo to confirm sync

## Potential Conflict Areas
Files that might conflict (both you and Karpathy may have modified):
- `README.md`
- `scripts/*.py`
- `nanochat/*.py`
- `pyproject.toml`

If conflicts occur, resolve them file by file.
