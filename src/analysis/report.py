from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.database.database import Database
from src.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates various text and data reports from the database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def generate_text_report(self, person_id: str | None = None) -> str:
        """Generate a formatted text report per spec section 36."""
        lines = []
        
        if person_id:
            persons = [self.db.get_person_by_id(person_id)]
            persons = [p for p in persons if p]
        else:
            # Technically need a way to get all persons. 
            # We'll use a raw query if a helper isn't available, but we can query stories and get distinct persons.
            # For simplicity, if not provided, let's grab all persons that have stories.
            rows = self.db._conn.execute("SELECT * FROM persons ORDER BY last_seen DESC").fetchall()
            persons = [self.db._row_to_person(r) for r in rows]

        for p in persons:
            stories = self.db.get_stories_for_person(p.person_id)
            if not stories:
                continue

            lines.append(f"Person: @{p.current_username}")
            lines.append(f"Person ID: {p.person_id}")
            lines.append(f"Identity Status: {p.identity_status.value}")
            lines.append(f"Stories analyzed: {len(stories)}")
            lines.append("")

            for idx, s in enumerate(stories, start=1):
                lines.append(f"Story {idx}")
                lines.append(f"Story ID: {s.story_id}")
                date_str = s.detected_at.strftime("%Y-%m-%d")
                lines.append(f"Date: {date_str}")

                initial = self.db.get_initial_analysis(s.story_id)
                status_str = initial.sampling_decision.value if initial and initial.sampling_decision else "Unknown"
                lines.append(f"Status: {status_str.capitalize()}")

                final = self.db.get_final_analysis(s.story_id)
                if final:
                    summary = final.description or "No description."
                    conf = final.confidence if final.confidence is not None else 0.0
                    lines.append(f"Summary: {summary}")
                    lines.append(f"Confidence: {conf:.2f}")
                else:
                    lines.append("Summary: Not finalized.")
                    lines.append("Confidence: N/A")
                
                lines.append("")
                
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines).strip()

    def export_json(self, output_path: str | Path, person_id: str | None = None) -> Path:
        """Export analyzed stories to a JSON file."""
        out_path = Path(output_path)
        
        if person_id:
            rows = self.db._conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,)).fetchall()
        else:
            rows = self.db._conn.execute("SELECT * FROM persons").fetchall()

        data = {"persons": []}
        
        for r in rows:
            p = self.db._row_to_person(r)
            stories = self.db.get_stories_for_person(p.person_id)
            if not stories:
                continue
                
            p_data = {
                "person_id": p.person_id,
                "username": p.current_username,
                "stories": []
            }
            
            for s in stories:
                final = self.db.get_final_analysis(s.story_id)
                s_dict = {
                    "story_id": s.story_id,
                    "date": s.detected_at.isoformat(),
                    "finalized": bool(final)
                }
                if final:
                    s_dict.update({
                        "content_type": final.content_type,
                        "description": final.description,
                        "confidence": final.confidence,
                    })
                p_data["stories"].append(s_dict)
                
            data["persons"].append(p_data)
            
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        logger.info("Exported JSON report to %s", out_path)
        return out_path

    def export_csv(self, output_path: str | Path) -> Path:
        """Export a flat CSV of all analyzed stories."""
        out_path = Path(output_path)
        
        rows = self.db._conn.execute("SELECT * FROM persons").fetchall()
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "username", "story_id", "content_type", "description", "confidence", "date"])
            
            for r in rows:
                p = self.db._row_to_person(r)
                stories = self.db.get_stories_for_person(p.person_id)
                for s in stories:
                    final = self.db.get_final_analysis(s.story_id)
                    if final:
                        writer.writerow([
                            p.person_id,
                            p.current_username,
                            s.story_id,
                            final.content_type or "",
                            final.description or "",
                            f"{final.confidence:.2f}" if final.confidence is not None else "",
                            s.detected_at.isoformat()
                        ])
                        
        logger.info("Exported CSV report to %s", out_path)
        return out_path

    def export_raw_json(self, output_dir: str | Path) -> Path:
        """Export raw AI outputs organized by profile and story to a timestamped JSON file."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{timestamp}_raw.json"
        
        data = {}
        rows = self.db._conn.execute("SELECT * FROM persons").fetchall()
        for r in rows:
            p = self.db._row_to_person(r)
            p_data = {
                "person_id": p.person_id,
                "username": p.current_username,
                "stories": {}
            }
            
            stories = self.db.get_stories_for_person(p.person_id)
            for s in stories:
                s_data = {
                    "story_id": s.story_id,
                    "date": s.detected_at.isoformat(),
                    "initial_analysis_raw": None,
                    "final_analysis_raw": None
                }
                
                # We can grab the JSON strings directly from the db
                init_row = self.db._conn.execute("SELECT * FROM initial_analyses WHERE story_id = ?", (s.story_id,)).fetchone()
                if init_row:
                    s_data["initial_analysis_raw"] = {k: init_row[k] for k in init_row.keys()}
                    
                fin_row = self.db._conn.execute("SELECT * FROM final_analyses WHERE story_id = ? ORDER BY created_at DESC LIMIT 1", (s.story_id,)).fetchone()
                if fin_row:
                    s_data["final_analysis_raw"] = {k: fin_row[k] for k in fin_row.keys()}
                    
                p_data["stories"][s.story_id] = s_data
                
            data[p.current_username] = p_data
            
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        logger.info("Exported raw output JSON to %s", out_path)
        return out_path
