import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DIST_DIR = ROOT_DIR / "dist"

# Repository name for GitHub Pages base path
REPO_NAME = os.getenv("GITHUB_REPOSITORY", "").split("/")[-1] or ROOT_DIR.name
BASE_URL = f"/{REPO_NAME}/"


def extract_title(md_file: Path) -> str:
    """Extract presentation title from frontmatter or first heading."""
    content = md_file.read_text(encoding="utf-8")
    # Check frontmatter title: "..."
    match = re.search(r"^title:\s*[\"']?(.*?)[\"']?$", content, re.MULTILINE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    # Check first # Heading
    match_h1 = re.search(r"^#\s+(.*?)$", content, re.MULTILINE)
    if match_h1:
        return match_h1.group(1).strip()
    return md_file.stem.replace("slides-", "").replace("slides", "").replace("-", " ").title() or "Presentation"


def build_all_decks() -> None:
    """Find all slides*.md files in docs/ and build each into dist/<slug>/."""
    slide_files = sorted(DOCS_DIR.glob("slides*.md"))
    if not slide_files:
        print("❌ No slides*.md files found in docs/")
        sys.exit(1)

    print(f"🚀 Found {len(slide_files)} slide deck(s) to build for repo '{REPO_NAME}':")
    for f in slide_files:
        print(f"   • {f.name}")

    # Ensure clean dist
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    decks_info = []

    for md_file in slide_files:
        # Determine slug/sub-route name
        # e.g., slides-benchmark.md -> benchmark
        # e.g., slides-executive.md -> executive
        # e.g., slides.md -> default
        raw_slug = md_file.stem.replace("slides-", "")
        if raw_slug == "slides":
            slug = "default"
        else:
            slug = raw_slug

        deck_title = extract_title(md_file)
        deck_dist = DIST_DIR / slug
        deck_base = f"{BASE_URL}{slug}/"

        print(f"\n📦 Building [{deck_title}] -> {deck_base} (out: dist/{slug})")

        cmd = [
            "npx",
            "-y",
            "@slidev/cli",
            "build",
            str(md_file),
            "--base",
            deck_base,
            "--out",
            str(deck_dist),
        ]

        try:
            subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
            decks_info.append({"slug": slug, "title": deck_title, "path": f"{slug}/", "file": md_file.name})
            print(f"✅ Successfully built '{slug}'")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed building '{md_file.name}': {e}")
            sys.exit(1)

    # Generate landing index.html portal at root of dist/
    generate_portal(decks_info)


def generate_portal(decks: list[dict[str, str]]) -> None:
    """Generate an executive Atos-styled portal index.html linking to all presentations."""
    portal_file = DIST_DIR / "index.html"

    cards_html = ""
    for deck in decks:
        cards_html += f"""
      <a href="{deck['path']}" class="deck-card">
        <div class="deck-tag">Presentation</div>
        <div class="deck-title">{deck['title']}</div>
        <div class="deck-desc">Source: <code>docs/{deck['file']}</code></div>
        <div class="deck-action">Launch Slides ➔</div>
      </a>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{REPO_NAME.replace('-', ' ').title()} - Slide Decks Portal</title>
  <style>
    :root {{
      --atos-navy: #00005b;
      --atos-blue: #0073e6;
      --atos-cyan: #43c7f4;
      --atos-bg: #f5f7fc;
      --atos-text: #161650;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
      background-color: var(--atos-bg);
      color: var(--atos-text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: var(--atos-navy);
      color: white;
      padding: 2.5rem 3rem;
      border-bottom: 4px solid var(--atos-blue);
    }}
    header h1 {{
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }}
    header p {{
      color: var(--atos-cyan);
      font-size: 1.1rem;
    }}
    main {{
      flex: 1;
      padding: 3rem;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
    }}
    .section-title {{
      font-size: 1.4rem;
      color: var(--atos-navy);
      margin-bottom: 1.5rem;
      font-weight: 700;
      border-left: 4px solid var(--atos-blue);
      padding-left: 0.8rem;
    }}
    .decks-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1.8rem;
    }}
    .deck-card {{
      background: white;
      border-radius: 8px;
      padding: 1.8rem;
      text-decoration: none;
      color: inherit;
      border: 1px solid #e1e6f0;
      box-shadow: 0 4px 12px rgba(0,0,91,0.04);
      transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
      display: flex;
      flex-direction: column;
    }}
    .deck-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 8px 24px rgba(0,115,230,0.12);
      border-color: var(--atos-blue);
    }}
    .deck-tag {{
      display: inline-block;
      align-self: flex-start;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--atos-blue);
      background: #eef6ff;
      padding: 0.3rem 0.6rem;
      border-radius: 4px;
      margin-bottom: 0.8rem;
    }}
    .deck-title {{
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--atos-navy);
      margin-bottom: 0.6rem;
      line-height: 1.3;
    }}
    .deck-desc {{
      font-size: 0.85rem;
      color: #6a7495;
      margin-bottom: 1.5rem;
      flex: 1;
    }}
    .deck-desc code {{
      background: #f0f2f8;
      padding: 2px 5px;
      border-radius: 3px;
      color: var(--atos-navy);
    }}
    .deck-action {{
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--atos-blue);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    footer {{
      text-align: center;
      padding: 2rem;
      color: #8892b0;
      font-size: 0.85rem;
      border-top: 1px solid #e1e6f0;
      background: white;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{REPO_NAME.replace('-', ' ').title()} Presentations</h1>
    <p>Interactive Atos Slide Decks Portal</p>
  </header>
  <main>
    <div class="section-title">Available Presentations ({len(decks)})</div>
    <div class="decks-grid">
      {cards_html}
    </div>
  </main>
  <footer>
    © Atos Group - Corporate & Technical Presentations Portal
  </footer>
</body>
</html>
"""
    portal_file.write_text(html, encoding="utf-8")
    print(f"\n🌐 Generated portal landing page at {portal_file}")


if __name__ == "__main__":
    build_all_decks()
