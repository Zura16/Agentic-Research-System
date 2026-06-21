#!/usr/bin/env python3
import os
import sys
import json
import math
import subprocess

STATE_FILE = ".git_progressive_push_state.json"
IGNORED_DIRS = {
    ".git", "venv", "node_modules", ".pytest_cache", 
    "__pycache__", "dist", "vector_store", "test_vector_store", 
    "test_uploads", "uploads", ".gemini"
}
IGNORED_FILES = {
    STATE_FILE, ".DS_Store"
}

def run_command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Error running command {' '.join(args)}:\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout.strip()

def get_all_project_files(repo_dir):
    """Recursively crawls the workspace to list all files that aren't ignored."""
    project_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        for file in files:
            if file in IGNORED_FILES or file.endswith(".pyc"):
                continue
            # Get path relative to the repo directory
            rel_path = os.path.relpath(os.path.join(root, file), repo_dir)
            project_files.append(rel_path)
            
    return sorted(project_files)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Failed to parse state file, recreating it: {e}")
    return {"pushed_files": [], "all_files": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)

    print(f"[*] Running 15% daily progressive push inside: {repo_dir}")

    # 1. Check if git repo is initialized
    if not os.path.exists(".git"):
        print("[!] Not a git repository. Initializing local git...")
        run_command(["git", "init"])
        # Create standard branch 'main'
        run_command(["git", "checkout", "-b", "main"])
        
    # 2. Check remote repository
    remote_info = run_command(["git", "remote", "-v"])
    if not remote_info:
        print("[!] No git remote found.")
        print("[*] Checking GitHub CLI credentials...")
        gh_auth = run_command(["env", "-u", "GITHUB_TOKEN", "gh", "auth", "status"])
        
        if gh_auth and "Logged in to github.com" in gh_auth:
            print("[*] Creating a new private repository on GitHub...")
            repo_name = os.path.basename(repo_dir).lower().replace(" ", "-")
            create_repo_res = run_command([
                "gh", "repo", "create", repo_name, 
                "--private", "--source=.", "--remote=origin"
            ])
            if create_repo_res:
                print(f"[+] Repository '{repo_name}' created and set as remote 'origin'.")
            else:
                print("[!] Failed to create repository automatically via GitHub CLI.")
                sys.exit(1)
        else:
            print("[!] Please authenticate with GitHub CLI ('gh auth login') or manually add a remote:")
            print("    git remote add origin <your-github-repo-url>")
            sys.exit(1)

    # 3. Read state
    state = load_state()
    all_files = get_all_project_files(repo_dir)
    pushed_files = state.get("pushed_files", [])

    # Filter out files that no longer exist
    pushed_files = [f for f in pushed_files if os.path.exists(f)]

    # Identify remaining files
    remaining_files = [f for f in all_files if f not in pushed_files]

    if not remaining_files:
        print("[*] All files have already been progressive-pushed! Project upload complete.")
        sys.exit(0)

    # Calculate 15% of total files (at least 1 file)
    total_count = len(all_files)
    chunk_size = math.ceil(total_count * 0.15)
    files_to_push = remaining_files[:chunk_size]

    print(f"[*] Total files in project: {total_count}")
    print(f"[*] Already pushed: {len(pushed_files)} ({len(pushed_files)/total_count*100:.1f}%)")
    print(f"[*] Staging next 15% ({len(files_to_push)} file(s)) of remaining {len(remaining_files)} files:")

    # 4. Stage files
    for file_path in files_to_push:
        print(f"    [+] git add {file_path}")
        run_command(["git", "add", file_path])

    # 5. Commit
    basenames = [os.path.basename(f) for f in files_to_push]
    commit_msg = f"Progressive push: added {', '.join(basenames)}"
    if len(commit_msg) > 75:
        commit_msg = f"Progressive push: added {len(files_to_push)} file(s)"
        
    print(f"[*] Committing: '{commit_msg}'")
    commit_res = run_command(["git", "commit", "-m", commit_msg])
    if commit_res is None:
        print("[!] Git commit failed.")
        sys.exit(1)

    # 6. Push
    print("[*] Pushing to remote...")
    # Get active branch name
    branch = run_command(["git", "branch", "--show-current"]) or "main"
    push_res = run_command(["git", "push", "-u", "origin", branch])
    if push_res is None:
        print("[!] Git push failed.")
        sys.exit(1)

    # 7. Update State
    state["pushed_files"] = pushed_files + files_to_push
    state["all_files"] = all_files
    save_state(state)

    print(f"[+] Progressive push complete! Total progress: {len(state['pushed_files'])}/{total_count} files pushed.")

if __name__ == "__main__":
    main()
