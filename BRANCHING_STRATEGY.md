# Git Branching Strategy - OC6 LinkedIn Lead Scoring

This project follows **Git Flow** with semantic versioning.

---

## 📊 Branch Structure

```
main (production)
├── v0.1.0 ← current release
├── v0.1.1 ← hotfixes
├── v1.0.0 ← future major release
│
develop (integration)
├── feature/db-implementation
├── feature/shap-endpoint
└── feature/api-auth
│
release/1.0.0 (release preparation)
└── feature/database-integration
│
hotfix/critical-bug
```

---

## 🏷️ Semantic Versioning

**Format:** `MAJOR.MINOR.PATCH` (e.g., v1.2.3)

- **PATCH (0.1.X)**: Bug fixes, hotfixes, no new functionality
  - Example: v0.1.1, v0.1.2
  - Source: `hotfix/*` branches

- **MINOR (0.Y.0)**: New features, backward compatible
  - Example: v0.2.0, v0.3.0
  - Source: `feature/*` branches via `develop`

- **MAJOR (X.0.0)**: Breaking changes, major features
  - Example: v1.0.0, v2.0.0
  - Source: `release/*` branches

---

## 🌿 Branch Types

### 1. `main` - Production Branch
- **Purpose:** Always reflects production-ready state
- **Protected:** Yes (require PR reviews)
- **Tagged:** Every merge gets a version tag
- **Deployed:** Automatically to HF Spaces via CI/CD

### 2. `develop` - Integration Branch
- **Purpose:** Integration branch for ongoing development
- **Source:** Branched from `main` at v0.1.0
- **Merges from:** `feature/*` branches
- **Merges to:** `release/*` branches
- **Protected:** Yes (require PR reviews)

### 3. `release/*` - Release Preparation
- **Purpose:** Prepare a new production release
- **Naming:** `release/X.Y.0` (e.g., `release/1.0.0`)
- **Source:** Branched from `develop`
- **Contains:** Feature branches for that release
- **Testing:** Comprehensive testing happens here
- **Merges to:** `main` (then tag) + back to `develop`

**Example workflow:**
```bash
# Create release branch
git checkout develop
git checkout -b release/1.0.0

# Feature work
git checkout -b feature/database-integration release/1.0.0
# ... work ...
git checkout release/1.0.0
git merge --no-ff feature/database-integration

# When ready
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff main
```

### 4. `feature/*` - Feature Development
- **Purpose:** Develop new features
- **Naming:** `feature/descriptive-name`
- **Source:** Branched from `develop` or `release/*`
- **Merges to:** `develop` or `release/*`
- **Lifetime:** Deleted after merge

**Examples:**
- `feature/database-integration`
- `feature/shap-endpoint`
- `feature/api-authentication`

### 5. `hotfix/*` - Emergency Fixes
- **Purpose:** Urgent production bug fixes
- **Naming:** `hotfix/issue-description`
- **Source:** Branched from `main`
- **Merges to:** `main` + `develop`
- **Version:** Increments PATCH (v0.1.X → v0.1.Y)

**Example workflow:**
```bash
# Create hotfix
git checkout main
git checkout -b hotfix/fix-prediction-error

# Fix the bug
# ... work ...

# Merge to main
git checkout main
git merge --no-ff hotfix/fix-prediction-error
git tag -a v0.1.1 -m "Hotfix v0.1.1: Fix prediction error"
git push origin main --tags

# Merge to develop
git checkout develop
git merge --no-ff hotfix/fix-prediction-error

# Delete hotfix branch
git branch -d hotfix/fix-prediction-error
```

---

## 🔄 Typical Workflows

### Adding a New Feature

```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature

# 2. Work on feature
# ... commits ...

# 3. Push and create PR to develop
git push -u origin feature/my-new-feature
gh pr create --base develop --title "feat: My new feature"

# 4. After review and merge
git checkout develop
git pull origin develop
git branch -d feature/my-new-feature
```

### Preparing a Release

```bash
# 1. Create release branch from develop
git checkout develop
git checkout -b release/1.1.0
git push -u origin release/1.1.0

# 2. Feature work happens in release branch
# ... testing, bug fixes ...

# 3. When ready, merge to main
git checkout main
git merge --no-ff release/1.1.0
git tag -a v1.1.0 -m "Release v1.1.0: Description"
git push origin main --tags

# 4. Merge back to develop
git checkout develop
git merge --no-ff main
git push origin develop

# 5. Delete release branch
git branch -d release/1.1.0
git push origin --delete release/1.1.0
```

### Emergency Hotfix

```bash
# 1. Create hotfix from main
git checkout main
git checkout -b hotfix/critical-bug

# 2. Fix the issue
# ... commits ...

# 3. Merge to main and tag
git checkout main
git merge --no-ff hotfix/critical-bug
git tag -a v0.1.2 -m "Hotfix v0.1.2: Fix critical bug"
git push origin main --tags

# 4. Merge to develop
git checkout develop
git merge --no-ff hotfix/critical-bug
git push origin develop

# 5. Cleanup
git branch -d hotfix/critical-bug
```

---

## 📋 Version History

| Version | Date | Type | Description |
|---------|------|------|-------------|
| v0.1.0 | 2025-11-19 | Initial | Initial setup: FastAPI + MLflow + Tests |
| v1.0.0 | TBD | Major | LemList integration + Model deployment |

---

## 🔒 Branch Protection Rules

### `main` branch:
- ✅ Require pull request reviews (1 minimum)
- ✅ Require status checks to pass (CI/CD)
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

### `develop` branch:
- ✅ Require pull request reviews (1 minimum)
- ✅ Require status checks to pass
- ⚠️ Allow force pushes (for rebasing)

---

## 📚 References

- [Semantic Versioning](https://semver.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

*Last updated: 2025-11-19*
