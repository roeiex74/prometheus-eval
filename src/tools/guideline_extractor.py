import os
import re
import argparse
from typing import Dict, List, Set

class GuidelineExtractor:
    def __init__(self, text_path: str):
        self.text_path = text_path
        self.raw_text = self._load_text()
        self.chapters = self._parse_chapters()
        
        # Mapping Agent -> Keywords/Section identifiers
        self.agent_mapping = {
            "Project_Architect": ["13", "3", "15"], # Chapters 13, 3, 15
            "Documentation_Agent": ["4", "9"], # Chapters 4, 9 (Git/Docs)
            "Security_Agent": ["5"], # Chapter 5
            "QA_Agent": ["6", "12", "13"], # QA, Standards, Checklist
            "Research_Agent": ["7"], # Research
            "UX_Agent": ["8"], # UX
        }
        
    def _load_text(self) -> List[str]:
        if not os.path.exists(self.text_path):
            return []
        with open(self.text_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _parse_chapters(self) -> Dict[str, str]:
        """
        Parses the text into chapters.
        Skips headers/TOC (assumes content starts after Page 4).
        Identifies chapters by lines starting with 'Number'.
        Handles cases where space is missing between number and Hebrew text.
        Enforces ordered chapter numbers (Max 20) to avoid capturing references as chapters.
        Ignores footers containing copyright symbols.
        """
        chapters = {}
        current_chapter_id = None
        current_lines = []
        max_chapter_seen = 0
        
        start_parsing = False
        
        # Regex for page markers
        page_pattern = re.compile(r"--- Page (\d+) ---")
        # Regex for chapter headers: Start of line, Number. 
        # We don't enforce space because of Hebrew PDF artifacts (e.g., "3Title").
        chapter_start_pattern = re.compile(r"^(\d+)")
        
        for line in self.raw_text:
            text_line = line.strip()
            
            # Check for page markers
            page_match = page_pattern.search(text_line)
            if page_match:
                page_num = int(page_match.group(1))
                if page_num >= 5:
                    start_parsing = True
                continue
            
            if not start_parsing:
                continue

            # Ignore footers/copyright lines
            if "©" in text_line or "Dr. Segal" in text_line:
                continue

            # Check for Chapter Header
            match = chapter_start_pattern.match(text_line)
            if match:
                try:
                    num_str = match.group(1)
                    num = int(num_str)
                    
                    # Verify it's not a subchapter (e.g., "1.1")
                    # Check the character immediately following the number
                    is_subchapter = False
                    if len(text_line) > len(num_str):
                        next_char = text_line[len(num_str)]
                        if next_char == '.':
                            is_subchapter = True
                    
                    if not is_subchapter:
                        # Max chapter is 20 (based on TOC ending at 19 References)
                        # This prevents "21..." references from becoming chapters.
                        if 1 <= num <= 20: 
                            if num > max_chapter_seen:
                                if current_chapter_id:
                                    chapters[current_chapter_id] = "\n".join(current_lines)
                                
                                current_chapter_id = str(num)
                                current_lines = [text_line]
                                max_chapter_seen = num
                                continue
                except ValueError:
                    pass

            if current_chapter_id:
                current_lines.append(text_line)
        
        # Capture last chapter
        if current_chapter_id:
            chapters[current_chapter_id] = "\n".join(current_lines)
            
        return chapters

    def get_chapter(self, chapter_id: str) -> str:
        """Returns the full text of a specific chapter."""
        return self.chapters.get(str(chapter_id), "")

    def extract_context(self) -> Dict[str, str]:
        extracted_data = {}
        
        for agent, chapter_ids in self.agent_mapping.items():
            content = []
            
            for cid in chapter_ids:
                chapter_text = self.get_chapter(cid)
                if chapter_text:
                    content.append(f"--- Chapter {cid} ---\n{chapter_text}")
                else:
                    content.append(f"Chapter {cid} not found or empty.")
            
            extracted_data[agent] = "\n\n".join(content)
            
        return extracted_data

    def save_agents_md(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        data = self.extract_context()
        for agent, context in data.items():
            filename = f"{agent}.md"
            path = os.path.join(output_dir, filename)
            
            md_content = f"""# {agent} - Guidelines

## Relevant Chapters
The following content is extracted from the submission guidelines based on chapters relevant to this agent.

```text
{context}
```
"""
            with open(path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"Generated {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract guidelines chapters.")
    parser.add_argument("--file", default="guidelines_text.txt", help="Path to guidelines text file")
    parser.add_argument("--chapters", nargs="+", help="List of chapter numbers to extract (e.g. 1 3 13)")
    parser.add_argument("--agent", help="Extract for a specific agent (Project_Architect, QA_Agent, etc.)")
    parser.add_argument("--save-all", action="store_true", help="Generate all agent MD files")
    
    args = parser.parse_args()
    
    extractor = GuidelineExtractor(args.file)
    
    if args.chapters:
        for ch in args.chapters:
            print(f"\n=== Chapter {ch} ===\n")
            print(extractor.get_chapter(ch))
            
    elif args.agent:
        contexts = extractor.extract_context()
        if args.agent in contexts:
            print(f"\n=== {args.agent} Guidelines ===\n")
            print(contexts[args.agent])
        else:
            print(f"Agent {args.agent} not found. Available: {list(contexts.keys())}")
            
    elif args.save_all:
        extractor.save_agents_md("agent_docs")
        
    else:
        sorted_keys = sorted(list(extractor.chapters.keys()), key=lambda x: int(x))
        print(f"Loaded {len(extractor.chapters)} chapters: {sorted_keys}")
        print("Use --chapters [X Y] to print content or --save-all to regenerate docs.")
