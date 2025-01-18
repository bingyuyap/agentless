Ok this is a draft of the new README for running agentless on a local repo

### Setting Up Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Upgrade pip and setuptools:
   ```bash
   pip install --upgrade pip setuptools
   ```


### Installing Dependencies

1. Install requirements from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
   
the localization command would be:

```bash
python -m agentless.fl.localize \
    --local_repo /path/to/repo/bugs.json \
    --local_repo_path /path/to/repo \
    --output_folder ./results/file_level \
    --file_level \
    --related_level
```

localize irrelevant

```bash
python -m agentless.fl.localize \
    --local_repo /path/to/your/bugs.json \  # Path to your local bugs JSON
    --local_repo_path /path/to/your/repo \  # Path to your local repository
    --file_level \
    --irrelevant \
    --output_folder results/file_level_irrelevant \
    --num_threads 10 \
    --skip_existing
```
