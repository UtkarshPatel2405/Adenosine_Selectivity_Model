# src/cleanup_outputs.py
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def archive_old_outputs():
    """
    Moves all old run-ID barcoded outputs (starting with 'ADENO_') into outputs/archive/
    to keep the active directories clean and prevent clutter.
    """
    from src.config import OUTPUTS_DIR
    
    archive_dir = OUTPUTS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Archiving old outputs in %s...", OUTPUTS_DIR)
    
    # Files to look for in outputs/ and its subdirectories
    count = 0
    # Walk through outputs/ but skip outputs/archive/
    for path in OUTPUTS_DIR.rglob("*"):
        if archive_dir in path.parents or path == archive_dir:
            continue
        
        # We only want to archive files, and specifically those starting with "ADENO_"
        if path.is_file() and path.name.startswith("ADENO_"):
            # Determine target archive path, maintaining relative structure if possible
            rel_path = path.relative_to(OUTPUTS_DIR)
            dest_path = archive_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.move(str(path), str(dest_path))
                count += 1
            except Exception as e:
                logger.error("Failed to archive file %s: %s", path, e)
                
    logger.info("Archived %d old run-barcoded files into %s", count, archive_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    archive_old_outputs()
