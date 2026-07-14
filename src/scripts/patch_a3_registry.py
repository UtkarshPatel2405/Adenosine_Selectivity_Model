import json
import logging
import requests
from pathlib import Path
from src.pdb_utils import pdb_to_smiles

logger = logging.getLogger(__name__)

def patch_a3():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Patching A3 registry...")
    
    uniprot_id = "P33765"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 100
            }
        }
    }
    
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=20)
        r.raise_for_status()
        result_set = r.json().get("result_set", [])
        pdb_ids = [x["identifier"] for x in result_set]
        logger.info(f"Found {len(pdb_ids)} PDB IDs for subtype A3.")
        
        a3_entries = []
        for pdb_id in pdb_ids:
            logger.info(f"  Fetching ligands for PDB ID {pdb_id}...")
            ligands = pdb_to_smiles(pdb_id)
            real_ligands = []
            for lig in ligands:
                if lig["ccd"] in ("HOH", "DOD", "GOL", "SO4", "PO4", "NA", "CL", "EDT", "ACT", "DMS", "PEG", "EDO", "FMT", "BU1"):
                    continue
                if lig["smiles"]:
                    real_ligands.append(lig)
            if real_ligands:
                a3_entries.append({
                    "pdb_id": pdb_id,
                    "ligands": real_ligands
                })
        
        # Load existing json
        json_path = Path("data/processed/adenosine_pdb_ligands.json")
        if json_path.exists():
            with open(json_path) as f:
                registry = json.load(f)
        else:
            registry = {}
            
        registry["A3"] = a3_entries
        
        with open(json_path, "w") as f:
            json.dump(registry, f, indent=2)
            
        logger.info("Successfully patched A3 registry!")
        
    except Exception as e:
        logger.error(f"Failed to patch A3: {e}")

if __name__ == "__main__":
    patch_a3()
