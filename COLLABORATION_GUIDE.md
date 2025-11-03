# Team Collaboration Guide

**Git Workflow & Best Practices for 3-Person Team**

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Git Workflow](#git-workflow)
3. [Branch Strategy](#branch-strategy)
4. [Commit Guidelines](#commit-guidelines)
5. [Handling Conflicts](#handling-conflicts)
6. [Code Review Process](#code-review-process)
7. [Safety Rules](#safety-rules)

---

## Quick Reference

### Daily Workflow

```bash
# ============ START WORK ============
# 1. Make sure you're on main and up-to-date
git checkout main
git pull origin main

# 2. Create your feature branch
git checkout -b feature/your-feature-name

# ============ DURING WORK ============
# 3. Make changes, test locally

# 4. Check what changed
git status
git diff

# 5. Stage and commit
git add <files>
git commit -m "Clear description of what you did"

# 6. Push to GitHub
git push origin feature/your-feature-name

# ============ FINISH WORK ============
# 7. Create Pull Request on GitHub
# 8. Wait for teammate review
# 9. After approval, merge via GitHub
# 10. Delete branch and update local
git checkout main
git pull origin main
git branch -d feature/your-feature-name
```

---

## Git Workflow

### Overview

We use **Feature Branch Workflow**:
- `main` branch = stable, working code
- Feature branches = work in progress
- Pull Requests = code review before merging

```
main branch:     A---B---C---F---G
                      \     /
feature branch:        D---E
                      (work) (merge)
```

### Step-by-Step Workflow

#### Step 1: Start a New Feature

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create feature branch with descriptive name
git checkout -b feature/pareto-visualization
# or
git checkout -b fix/distance-calculation-bug
# or
git checkout -b docs/add-model-explanation
```

**Branch naming**:
- `feature/description` - new functionality
- `fix/description` - bug fixes
- `docs/description` - documentation
- `refactor/description` - code improvements

#### Step 2: Work on Your Feature

```bash
# Make changes to files

# Check what you changed
git status          # See modified files
git diff            # See line-by-line changes

# Test your code locally!
python pfizer_optimization.py  # or relevant test
```

#### Step 3: Commit Your Changes

```bash
# Stage specific files
git add pfizer_optimization.py
git add pareto_analysis.py

# Or stage all changes (use carefully!)
git add .

# Commit with clear message
git commit -m "Add epsilon-constraint method for Pareto frontier"

# Push to GitHub
git push origin feature/pareto-visualization
```

**First time pushing a new branch?**
```bash
git push -u origin feature/your-branch-name
```

#### Step 4: Create Pull Request (PR)

1. Go to GitHub repository
2. Click "Pull requests" → "New pull request"
3. Select your branch
4. Fill in:
   - **Title**: Clear, concise summary
   - **Description**: What you changed and why
   - **Reviewers**: Assign your teammates
5. Click "Create pull request"

#### Step 5: Code Review

**If you're the author**:
- Respond to comments
- Make requested changes
- Push updates (they auto-update the PR)

**If you're a reviewer**:
- Read the code carefully
- Test locally if needed:
  ```bash
  git fetch origin
  git checkout feature/teammate-branch
  python test_file.py
  ```
- Leave constructive comments
- Approve or request changes

#### Step 6: Merge and Clean Up

**After PR approved**:
1. Click "Merge pull request" on GitHub
2. Delete branch on GitHub (button appears after merge)
3. Update your local repo:
   ```bash
   git checkout main
   git pull origin main
   git branch -d feature/your-branch-name  # Delete local branch
   ```

---

## Branch Strategy

### Branch Types

```
main
├── feature/model-1-implementation
├── feature/model-2-implementation
├── feature/pareto-visualization
├── fix/workload-constraint-bug
└── docs/add-readme
```

### Protected Main Branch

**Rules for `main`**:
- ❌ Never commit directly to main
- ❌ Never push directly to main
- ✅ Always merge via Pull Request
- ✅ Must pass review before merge
- ✅ Should always be working/stable

**Why?**
- Prevents accidental breaking of working code
- Ensures code review
- Maintains project history
- Protects against mistakes

### Setting Up Branch Protection (One person does this on GitHub)

1. Go to GitHub repo → Settings → Branches
2. Add rule for `main`:
   - ✅ Require pull request before merging
   - ✅ Require approvals: 1
   - ✅ Dismiss stale approvals when new commits pushed
3. Save changes

---

## Commit Guidelines

### Good Commit Messages

**Format**:
```
Short summary (50 chars or less)

Optional detailed explanation of what changed and why.
Can be multiple lines.

- Bullet points for specific changes
- Reference issues if relevant
```

**Examples**:

✅ **Good**:
```
Add Model 2 disruption minimization

Implements the disruption objective using auxiliary variables
to linearize absolute value constraints. Tested on sample data.

- Added y variables for |x - A|
- Implemented two-sided constraints
- Updated documentation
```

✅ **Good** (simple):
```
Fix distance matrix indexing bug in load_data()
```

❌ **Bad**:
```
Update
```

❌ **Bad**:
```
Fixed some stuff and changed things
```

### What to Commit

✅ **Commit**:
- Source code (`.py`)
- Documentation (`.md`)
- Notebooks (`.ipynb`)
- Configuration files
- Data files (if small and essential)

❌ **Don't Commit**:
- Generated results (`.png`, `.csv`, `.pkl`)
- Cache files (`__pycache__/`)
- Virtual environment (`venv/`)
- IDE settings (`.vscode/`, `.idea/`)
- Log files (`.log`)

**Why?** The `.gitignore` file handles this automatically!

### Commit Frequency

**Good practice**:
- Commit **often** (every logical change)
- Each commit = one logical unit of work
- Don't wait until "everything is done"

**Example workflow**:
```bash
# Implement Model 1
git add pfizer_optimization.py
git commit -m "Implement Model 1 distance minimization"

# Add tests
git add test_model1.py
git commit -m "Add unit tests for Model 1"

# Update docs
git add README.md
git commit -m "Document Model 1 usage in README"
```

---

## Handling Conflicts

### What is a Merge Conflict?

Happens when:
- You and a teammate edited the same lines
- Git can't automatically merge

**Example**:
```
<<<<<<< HEAD (your changes)
def calculate_distance(x, y):
    return abs(x - y)
=======
def calculate_distance(x, y):
    return sqrt((x - y) ** 2)
>>>>>>> feature/teammate-branch (teammate's changes)
```

### Preventing Conflicts

1. **Communicate!** Tell team what files you're working on
2. **Pull often**: `git pull origin main` daily
3. **Keep branches short-lived**: merge within 1-2 days
4. **Divide work**: different files/functions per person

### Resolving Conflicts (Step-by-Step)

**Scenario**: You try to merge and get conflicts

```bash
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main  # ← Conflict!
```

**Resolution**:

1. **Don't panic!** Git shows which files have conflicts:
   ```
   CONFLICT (content): Merge conflict in pfizer_optimization.py
   ```

2. **Open the file** and look for conflict markers:
   ```python
   <<<<<<< HEAD
   # Your version
   result = x + y
   =======
   # Teammate's version
   result = x * y
   >>>>>>> main
   ```

3. **Decide what to keep**:
   - Keep your version
   - Keep teammate's version
   - Combine both
   - Write something new

4. **Remove conflict markers** and save:
   ```python
   # After resolution
   result = x + y  # Kept our version
   ```

5. **Mark as resolved**:
   ```bash
   git add pfizer_optimization.py
   git commit -m "Resolve merge conflict in calculate function"
   git push origin feature/your-branch
   ```

### When in Doubt

**Ask for help!**
- Call a quick team meeting
- Screen share and decide together
- Don't guess - wrong merges can break code

---

## Code Review Process

### Why Code Review?

- Catch bugs early
- Share knowledge
- Maintain code quality
- Learn from each other

### How to Review Code (Checklist)

**As a reviewer**:

```
□ Does the code run without errors?
□ Are the changes clear and well-commented?
□ Does it follow our coding style?
□ Are there any obvious bugs?
□ Does it solve the intended problem?
□ Are there tests (if needed)?
□ Is documentation updated?
```

**What to check**:
1. **Functionality**: Does it work?
2. **Correctness**: Is the logic right?
3. **Clarity**: Can you understand it?
4. **Style**: Does it match our conventions?

### Giving Feedback

✅ **Good feedback**:
```
"This function looks good! One suggestion: could you add a docstring 
explaining the parameters? Also, line 42 might throw an error if 
workload is empty - consider adding a check."
```

❌ **Bad feedback**:
```
"This is wrong."
```

### Responding to Feedback

**As author**:
- Don't take it personally - it's about the code!
- Ask questions if unclear
- Thank reviewers for their time
- Make requested changes or discuss why not

---

## Safety Rules

### The Golden Rules

1. **NEVER force push**: `git push --force` ❌ (destroys history)
2. **NEVER commit to main directly**: Always use branches ✅
3. **ALWAYS pull before starting work**: Stay updated ✅
4. **ALWAYS test before committing**: Make sure it works ✅
5. **ALWAYS review before merging**: Two pairs of eyes ✅

### Emergency: "I Messed Up!"

#### "I committed to main by accident!"

```bash
# Don't push! Undo last commit
git reset --soft HEAD~1

# Create proper branch
git checkout -b feature/my-feature

# Now commit properly
git commit -m "My changes"
git push origin feature/my-feature
```

#### "I pushed to main by accident!"

**Don't panic!**
1. Tell your team immediately
2. If no one has pulled yet:
   ```bash
   git revert HEAD  # Creates a new commit that undoes the last one
   git push origin main
   ```
3. If others pulled: Ask instructor/TA for help

#### "I deleted important work!"

```bash
# Find your lost commit
git reflog

# Recover it (use commit hash from reflog)
git checkout <commit-hash>
git checkout -b recovery-branch
```

#### "There are conflicts and I don't know what to do!"

```bash
# Abort the merge and get help
git merge --abort

# Or if during rebase
git rebase --abort

# Ask your teammates for help!
```

---

## Working Session Best Practices

### Before Starting Work

```bash
# Morning routine
git checkout main
git pull origin main
git checkout -b feature/today-task

# Announce to team (Slack/Discord/etc.)
"Working on Model 2 implementation today"
```

### During Work

- **Commit frequently** (every 30-60 minutes of work)
- **Pull from main** if working for several hours
- **Communicate** if changing shared files

### End of Session

```bash
# Push your work (even if unfinished)
git push origin feature/your-branch

# This backs up your work and lets teammates see progress
```

### End of Day/Week

- Create PR if feature is complete
- Update team on progress
- Note any blockers or questions

---

## Task Division Strategies

### Strategy 1: By Model

```
Person 1: Model 1 (minimize distance)
Person 2: Model 2 (minimize disruption)
Person 3: Epsilon-constraint & visualization
```

### Strategy 2: By Module

```
Person 1: Core optimization (pfizer_optimization.py)
Person 2: Analysis & visualization (pareto_analysis.py)
Person 3: Testing & documentation
```

### Strategy 3: By Phase

```
Week 1: Everyone - data loading and basic setup
Week 2: Split - model implementation
Week 3: Everyone - testing and debugging
Week 4: Split - visualization and documentation
```

**Key**: Communicate and adjust as needed!

---

## Communication Tips

### Daily Standup (5 min meeting or chat)

Each person shares:
1. What I did yesterday
2. What I'm doing today
3. Any blockers

**Example**:
```
Alice: "Finished Model 1 yesterday. Today working on tests. No blockers."
Bob: "Debugged distance matrix. Today starting Model 2. Need to understand 
     absolute value linearization - can someone explain?"
Charlie: "Worked on README. Today creating visualizations. No blockers."
```

### Use GitHub Issues

Create issues for:
- Bugs found
- Features to add
- Questions for team
- Tasks to assign

**Example**:
```
Title: "Implement workload balance constraint"
Assignee: Alice
Labels: feature, high-priority
Description: Add constraints ensuring each SR workload is between 0.8-1.2
```

---

## Git Commands Cheatsheet

```bash
# ===== BASIC =====
git status                  # See what changed
git diff                    # See specific changes
git log --oneline           # See commit history

# ===== BRANCHING =====
git branch                  # List branches
git checkout -b feature/x   # Create and switch to branch
git checkout main           # Switch to main
git branch -d feature/x     # Delete branch (after merge)

# ===== STAGING & COMMITTING =====
git add file.py             # Stage specific file
git add .                   # Stage all changes
git commit -m "message"     # Commit staged changes
git commit -am "message"    # Stage and commit (tracked files only)

# ===== SYNCING =====
git pull origin main        # Get latest from GitHub
git push origin branch      # Send your commits to GitHub
git fetch origin            # Download updates without merging

# ===== UNDOING =====
git checkout -- file.py     # Discard changes to file
git reset HEAD file.py      # Unstage file
git revert <commit>         # Undo a commit (safe)
git reset --soft HEAD~1     # Undo last commit, keep changes

# ===== MERGING =====
git merge main              # Merge main into current branch
git merge --abort           # Cancel merge if conflicts

# ===== INFORMATION =====
git remote -v               # See GitHub repo URL
git log --graph --oneline   # Visual commit history
```

---

## Troubleshooting Common Issues

### "Permission denied (publickey)"
```bash
# Set up SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"
# Add to GitHub: Settings → SSH keys
```

### "Your branch is behind 'origin/main'"
```bash
git pull origin main
```

### "Please commit your changes or stash them"
```bash
# Option 1: Commit
git add .
git commit -m "WIP: save current work"

# Option 2: Stash (temporary save)
git stash
git pull origin main
git stash pop
```

### "fatal: Not a git repository"
```bash
# Make sure you're in the right directory
cd /path/to/Pfizer_Global
```

---

## Resources

### Learn Git
- [Git Tutorial - Atlassian](https://www.atlassian.com/git/tutorials)
- [Git Cheatsheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [GitHub Guides](https://guides.github.com/)

### Interactive Practice
- [Learn Git Branching](https://learngitbranching.js.org/)
- [Git Exercises](https://gitexercises.fracz.com/)

### GitHub Help
- [Creating Pull Requests](https://docs.github.com/en/pull-requests)
- [Resolving Conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)

---

## Questions?

**Ask your teammates!** You're in this together. 🚀

---

**Last Updated**: November 2025
