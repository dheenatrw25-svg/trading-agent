from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGS_DIR = PROJECT_ROOT / "logs"
VAULT_DIR = PROJECT_ROOT / "vault"
PLAYBOOKS_DIR = VAULT_DIR / "01_Playbooks"
TRADE_MEMORIES_DIR = VAULT_DIR / "02_Trade_Memories"

ASSETS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PLAYBOOKS_DIR.mkdir(parents=True, exist_ok=True)
TRADE_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
