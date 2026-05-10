# Colab CLI — Training Playbook

## Assign GPU Server
```bash
colab-cli server assign --variant GPU --accelerator L4 --name "gridtokenx-train"
# GPU options: T4 (1.76 CCU/hr), L4 (4.82), G4 (2.00), A100 (11.77), H100 (14.43)
```

## Run Commands
```bash
# Shell commands go through bash -lc
colab-cli server run bash -lc 'echo hello && uname -a'
colab-cli server run python -V
colab-cli server run nvidia-smi

# Chain commands with bash -lc '...'
colab-cli server run bash -lc 'cd /content/gridtokenx && pip install -q -r requirements.txt'
```

## File Operations
```bash
# Upload (works well for small files, large files may fail with network errors)
colab-cli file upload local.txt /content/remote.txt

# Download entire directory as tarball
colab-cli server run bash -lc 'cd /content/gridtokenx && tar czf /tmp/results.tar.gz models/ data/processed/ results/'
# Then download via the Colab web UI or use file upload to move it somewhere accessible
```

## Server Management
```bash
colab-cli server ls                    # List assigned servers
colab-cli server ls --available        # Show available accelerators + CCU balance
colab-cli server info                  # Show current server details
colab-cli server ps                    # Live CPU/RAM/GPU stats
colab-cli server rm                    # Remove server (stop instance)
```

## Full Training Workflow

```bash
# 1. Assign server
colab-cli server assign --variant GPU --accelerator L4 --name "gridtokenx-train"

# 2. Clone repo
colab-cli server run bash -lc 'git clone https://github.com/NakaSato/gridtokenx-forecase-ml.git /content/gridtokenx'

# 3. Install deps
colab-cli server run bash -lc 'cd /content/gridtokenx && pip install -q -r requirements.txt'

# 4. Run full pipeline
colab-cli server run bash -lc 'cd /content/gridtokenx && python research/colab_train.py'

# 5. Download results
colab-cli server run bash -lc 'cd /content/gridtokenx && tar czf /tmp/results.tar.gz models/ data/processed/ results/'
# Then download /tmp/results.tar.gz from Colab UI or:
colab-cli file ls /tmp/results.tar.gz  # verify it exists
```

## Known Issues
- **Large file uploads fail**: `colab-cli file upload` fails for files >10MB with network errors. Use `git clone` on the remote instead.
- **`server run` is NOT a shell**: Commands are passed verbatim to the runtime. Use `bash -lc '...'` for complex commands with pipes, redirects, or &&.
- **Token expires ~59min**: If the session token expires, re-auth with `colab-cli auth`.
