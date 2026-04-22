---
name: sap-dropdown
description: Interact with SAP WebDynpro custom dropdowns inside iframes — click to open, scroll, and select items
browser_backend: direct
---

# SAP WebDynpro Custom Dropdowns

SAP WebDynpro forms do NOT use standard HTML `<select>` elements. Dropdowns are custom components
rendered as `<span>` or `<div>` elements. Clicking them opens a popup listbox overlay.
The popup items may require scrolling if the list is long.

**This skill applies whenever you need to select a value from a dropdown in SAP WebDynpro / SAP Fiori forms.**

## Key Concepts

- SAP forms are typically inside an **iframe** (e.g. `#application-Expense-claim-iframe`)
- All JavaScript must target the iframe's `contentDocument`, NOT the main page document
- Dropdown fields look like text inputs with a small arrow button next to them
- When clicked, they open a **popup listbox** (`[role="listbox"]` or class `lsListbox__items`)
- Items in the popup can be clicked to select them
- Long lists need **scrolling** inside the popup before the target item is visible

## How to Select a Dropdown Value

### Step 1 — Identify the iframe

```js
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
```

### Step 2 — Find and click the dropdown to open it

Dropdown fields in SAP WebDynpro have an arrow/button for expanding. Find the dropdown by its **label text** nearby, then click the arrow or the field itself:

```js
// Option A: Find by nearby label text, then click the dropdown trigger
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;

// Find all labels, locate the one matching the target
const labels = [...doc.querySelectorAll('[class*="lsLabel"], [class*="urLbl"]')];
const label = labels.find(l => l.textContent.includes('LABEL_TEXT'));

// The dropdown input is usually a sibling or nearby element
// Click the arrow button (small square next to the field)
const container = label.closest('[class*="lsControl"], tr, [class*="urCB"]');
const arrow = container.querySelector('[class*="arw"], [class*="Btn"], [id*="-btn"], [role="button"]');
if (arrow) arrow.click();
```

```js
// Option B: If you know the element ID (e.g. from a previous browser_read_page)
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const dropdown = doc.querySelector('#ELEMENT_ID');
dropdown.click();
```

### Step 3 — Wait for the popup to appear

After clicking, wait briefly for the popup listbox to render:

```
browser_wait timeout_ms=1000
```

### Step 4 — List available options (for discovery)

Use `browser_evaluate` to list all visible options in the open popup:

```js
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const items = [...doc.querySelectorAll('[role="option"], [class*="lsListbox__item"], [class*="urCBLI"]')];
JSON.stringify(items.map((it, i) => ({ index: i, text: it.textContent.trim(), id: it.id })));
```

### Step 5 — Scroll to the target item if needed

If the list is long, the target option might not be visible. Scroll the listbox popup:

```js
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;

// Find the scrollable container of the popup
const listbox = doc.querySelector('[role="listbox"], [class*="lsListbox__items"], [class*="urCBList"]');

// Scroll down step by step (each step ~200px)
listbox.scrollTop += 200;
```

After scrolling, list the options again to check if the target item is now visible.
Repeat scroll + list until you find the target option.

### Step 6 — Click the target option

```js
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const items = [...doc.querySelectorAll('[role="option"], [class*="lsListbox__item"], [class*="urCBLI"]')];
const target = items.find(it => it.textContent.trim().includes('TARGET_TEXT'));
if (target) {
  target.scrollIntoView({ block: 'center' });
  target.click();
}
```

### Step 7 — Verify the selection

After clicking, the popup should close and the dropdown field should show the selected value.
Use `browser_read_page` or `browser_evaluate` to confirm:

```js
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
// Read the current value of the dropdown field
const field = doc.querySelector('#DROPDOWN_ID');
field?.textContent?.trim() || field?.value;
```

## Complete Example: Select "Repas" from a dropdown labeled "Cat. frais"

```js
// 1. Open the dropdown
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const labels = [...doc.querySelectorAll('*')].filter(e => e.textContent.includes('Cat. frais') && e.children.length === 0);
const label = labels[0];
const row = label.closest('tr') || label.parentElement?.parentElement;
const arrow = row.querySelector('[role="button"], [class*="arw"], [class*="Btn"]');
if (arrow) arrow.click(); else row.querySelector('input, [role="combobox"]')?.click();
```

```
browser_wait timeout_ms=1000
```

```js
// 2. List options to see what's available
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const items = [...doc.querySelectorAll('[role="option"], [class*="lsListbox__item"], [class*="urCBLI"]')];
JSON.stringify(items.slice(0, 20).map((it, i) => ({ index: i, text: it.textContent.trim() })));
```

```js
// 3. If "Repas" is not visible, scroll down and list again
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const listbox = doc.querySelector('[role="listbox"], [class*="lsListbox"], [class*="urCBList"]');
if (listbox) listbox.scrollTop += 300;
```

```js
// 4. Click the target option
const iframe = document.querySelector('iframe[id*="iframe"]');
const doc = iframe.contentDocument;
const items = [...doc.querySelectorAll('[role="option"], [class*="lsListbox__item"], [class*="urCBLI"]')];
const target = items.find(it => it.textContent.trim().includes('Repas'));
if (target) { target.scrollIntoView({ block: 'center' }); target.click(); }
JSON.stringify({ found: !!target, text: target?.textContent?.trim() });
```

## Important Notes

- **Always re-query the iframe** in each `browser_evaluate` call — do NOT assume variables persist between calls
- After opening a dropdown, always **wait ~1 second** before listing items (popup render time)
- If the list has many items (>20), **scroll in increments** of 200-300px and re-list after each scroll
- Some SAP dropdowns use **combobox** pattern: you can also type into them to filter, then select
- If clicking the arrow doesn't work, try clicking directly on the text field/input area
- Popup overlays may appear **outside the normal DOM flow** — look for elements at the end of the `<body>` in the iframe
- After selecting, always verify the value was set correctly before moving on
