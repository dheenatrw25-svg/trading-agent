from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR

ASSETS_DIR = PROJECT_ROOT / "assets"
LOGS_DIR = PROJECT_ROOT / "logs"
VAULT_DIR = PROJECT_ROOT / "vault"
PLAYBOOKS_DIR = VAULT_DIR / "01_Playbooks"
MEMORIES_DIR = VAULT_DIR / "02_Trade_Memories"
TRADE_MEMORIES_DIR = MEMORIES_DIR

# Canonical vault playbook and memory files used by the forecasting pipeline.
MASTER_PLAYBOOK_FILE = PLAYBOOKS_DIR / "master_playbook.md"
PLAYBOOK_BACKUP_FILE = PLAYBOOKS_DIR / "master_playbook.backup.md"

ASSETS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
TRADE_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
