"""
extract_cookies.py — Extract Instagram cookies from your browser.

METHOD 1 (Automatic): Uses browser-cookie3 (may need admin on Windows)
    pip install browser-cookie3
    python extract_cookies.py

METHOD 2 (Manual — RECOMMENDED): Uses DevTools copy-paste
    python extract_cookies.py --manual

METHOD 3 (Console JS): Paste a JS snippet in DevTools console
    python extract_cookies.py --js
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "session.json"


def extract_auto(browser_name: str | None = None) -> list[dict]:
    """Extract cookies automatically using browser-cookie3."""
    try:
        import browser_cookie3
    except ImportError:
        print("ERROR: browser_cookie3 not installed. Run: pip install browser-cookie3")
        print("Or use --manual mode instead.")
        return []

    browsers = {
        "chrome": browser_cookie3.chrome,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
        "opera": browser_cookie3.opera,
        "brave": browser_cookie3.brave,
    }

    if browser_name:
        browser_fns = [(browser_name.lower(), browsers.get(browser_name.lower()))]
        if not browser_fns[0][1]:
            print(f"Unknown browser '{browser_name}'.")
            return []
    else:
        browser_fns = list(browsers.items())

    for name, fn in browser_fns:
        try:
            print(f"  Trying {name}...")
            cj = fn(domain_name=".instagram.com")
            cookies = []
            for c in cj:
                if "instagram" in c.domain:
                    cookies.append({
                        "name": c.name,
                        "value": c.value,
                        "domain": c.domain,
                        "path": c.path,
                        "secure": bool(c.secure),
                        "httpOnly": c.has_nonstandard_attr("HttpOnly"),
                        "sameSite": "Lax",
                    })
            if cookies:
                print(f"  Found {len(cookies)} cookies from {name}.")
                return cookies
        except Exception as e:
            print(f"  {name}: {e}")
    return []


def extract_manual() -> list[dict]:
    """Guide user through manual cookie extraction via DevTools."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MANUAL COOKIE EXTRACTION                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Open Instagram.com in Chrome/Edge                        ║
║  2. Make sure you are LOGGED IN                              ║
║  3. Press F12 to open DevTools                               ║
║  4. Go to the CONSOLE tab                                    ║
║  5. Paste this code and press Enter:                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

--- COPY THIS CODE (everything between the lines) ---
""")

    js_code = """JSON.stringify(document.cookie.split('; ').map(c => { const [name, ...rest] = c.split('='); return { name, value: rest.join('='), domain: '.instagram.com', path: '/', secure: true, httpOnly: false, sameSite: 'Lax' }; }))"""

    print(js_code)
    print("\n--- END OF CODE ---\n")
    print("6. Copy the output (starts with [ and ends with ])")
    print("7. Paste it below and press Enter:\n")

    user_input = input("Paste cookie JSON here > ").strip()

    if not user_input:
        print("No input provided.")
        return []

    try:
        cookies = json.loads(user_input)
        if isinstance(cookies, list) and len(cookies) > 0:
            # Mark httpOnly cookies (sessionid, csrftoken are httpOnly but
            # document.cookie can't access them — user needs Application tab)
            print(f"\nParsed {len(cookies)} cookies.")

            # Check if essential cookies are present
            names = {c["name"] for c in cookies}
            if "sessionid" not in names:
                print("\n⚠️  'sessionid' cookie is MISSING!")
                print("   document.cookie cannot access httpOnly cookies.")
                print("   Use the Application tab method instead:\n")
                return extract_application_tab()

            return cookies
        else:
            print("Invalid format. Expected a JSON array.")
            return []
    except json.JSONDecodeError:
        print("Could not parse as JSON. Make sure you copied the full output.")
        return []


def extract_application_tab() -> list[dict]:
    """Guide user to extract cookies from DevTools Application tab."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           APPLICATION TAB METHOD (Gets ALL cookies)          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Open Instagram.com in Chrome/Edge (logged in)            ║
║  2. Press F12 to open DevTools                               ║
║  3. Go to APPLICATION tab (not Console)                      ║
║  4. In the left sidebar: Storage → Cookies → instagram.com   ║
║  5. You will see a table of cookies                          ║
║                                                              ║
║  Copy these cookie values (right-click → Copy Value):        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    essential_cookies = ["sessionid", "csrftoken", "ds_user_id", "mid", "ig_did", "rur"]
    cookies = []

    for name in essential_cookies:
        value = input(f"  {name}: ").strip()
        if value:
            cookies.append({
                "name": name,
                "value": value,
                "domain": ".instagram.com",
                "path": "/",
                "secure": True,
                "httpOnly": name in ("sessionid",),
                "sameSite": "Lax",
            })

    if not cookies:
        print("No cookies entered.")
        return []

    print(f"\nCollected {len(cookies)} cookies.")
    return cookies


def save_cookies(cookies: list[dict], output_path: Path) -> None:
    """Save cookies to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2)

    names = {c["name"] for c in cookies}
    essential = {"sessionid", "csrftoken", "ds_user_id"}
    missing = essential - names

    print(f"\n✓ Saved {len(cookies)} cookies to {output_path}")

    if missing:
        print(f"⚠️  Missing essential cookies: {missing}")
    else:
        print(f"✓ All essential cookies present: {essential}")

    print(f"\nCookies saved ({len(cookies)}):")
    for c in cookies:
        v = c["value"][:8] + "..." if len(c["value"]) > 8 else c["value"]
        print(f"  {c['name']:20s} = {v}")

    print(f"\nNext steps:")
    print(f"  1. Upload {output_path.name} to your Colab notebook")
    print(f"  2. Run the notebook — Cell 6 will load these cookies")


def main():
    parser = argparse.ArgumentParser(description="Extract Instagram cookies")
    parser.add_argument("--browser", "-b", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--manual", "-m", action="store_true",
                        help="Use manual DevTools extraction")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_FILE

    print("=" * 50)
    print("  StoryZop Cookie Extractor")
    print("=" * 50)
    print()

    if args.manual:
        cookies = extract_application_tab()
    else:
        print("Attempting automatic extraction...")
        cookies = extract_auto(args.browser)

        if not cookies:
            print("\nAutomatic extraction failed.")
            print("Falling back to manual extraction...\n")
            cookies = extract_application_tab()

    if cookies:
        save_cookies(cookies, output_path)
    else:
        print("\nNo cookies extracted. Cannot continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()
