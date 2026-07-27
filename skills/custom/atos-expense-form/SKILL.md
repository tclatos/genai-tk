---
name: atos-expense-form
description: Fill out the Atos MyAtos Expense Claims form with expense details (dates, country, category, reason, and comments)
browser_backend: direct
---

# Atos MyAtos Expense Claims Form Filling

This skill guides you to fill out the Expense Claims form in the Atos MyAtos portal with expense details.

**Prerequisites:**
- User must already be authenticated on the Atos MyAtos portal
- The Expense Claims page must be open at `https://nextgen.myatos.net/sap/flp#Expense-claim`
- The "Create New Expense Report" button must have been clicked to open the form
- This skill does NOT handle authentication — use the `atos-portal` skill for login

## Form Structure

The Expense Claims form contains the following fields:

### Main Fields:
1. **Start Date** (Date de début) - Format: DD.MM.YYYY
2. **End Date** (Date de fin) - Format: DD.MM.YYYY
3. **Country** (Pays) - Dropdown selection
4. **Category** (Catégorie) - Dropdown selection (e.g., "Non refacturable" for non-billable)
5. **Reason** (Motif) - Text field for the expense reason
6. **Comments** (Commentaires) - Text area for additional comments
7. **Submit Button** (Interrompre/Submit) - Button to submit the form

## Navigation Flow

### Step 1 — Access the Form

```
1. browser_navigate to https://nextgen.myatos.net/sap/flp#Expense-claim
2. browser_wait with load_state="networkidle" (timeout: 5000ms)
3. Locate the "Créer nouveau décompte des frais" button (Create New Expense Report)
4. Click the button using JavaScript: 
   const iframe = document.querySelector('#application-Expense-claim-iframe');
   const button = iframe.contentDocument.querySelector('#WD5B');
   button.click();
5. browser_wait with timeout 3000ms for form to load
```

### Step 2 — Fill in the Start Date

```
1. Locate the start date field in the form
2. Enter the date in format DD.MM.YYYY (e.g., "17.04.2026" for April 17, 2026)
3. The field is typically the first date input in the form
```

### Step 3 — Fill in the End Date

```
1. Locate the end date field (usually right after the start date)
2. Enter the date in format DD.MM.YYYY (e.g., "17.04.2026" for April 17, 2026)
```

### Step 4 — Select Country

```
1. Locate the Country dropdown field
2. Select "France" from the dropdown options
3. Use browser_click or browser_evaluate to select the option
```

### Step 5 — Select Category

```
1. Locate the "Catégorie de déplacement (interne)" dropdown field
2. Select "Non Refacturable" (Non-billable) from the dropdown options
3. This indicates the expense is not billable to a client
```

### Step 6 — Fill in the Reason

```
1. Locate the Reason text field (Motif)
2. Enter the reason for the expense (e.g., "déplacement hackathon" for hackathon travel)
3. Use browser_type or browser_evaluate to fill the field
```

### Step 7 — Fill in the Comments

```
1. Locate the Comments text area (Commentaires)
2. Enter additional comments (e.g., "Test pour le hackathon" for hackathon test)
3. Use browser_type or browser_evaluate to fill the field
```

### Step 8 — Submit the Form

```
1. Locate the Submit button (typically labeled "Interrompre" or similar)
2. Click the button to submit the form
3. browser_wait with load_state="networkidle" (timeout: 5000ms) to confirm submission
```

## CSS Selectors and IDs

| Element | Selector/ID | Notes |
|---------|-------------|-------|
| Create Button | `#WD5B` | Inside iframe, click to open form |
| Start Date Field | Look for date input with label "Date de début" | Format: DD.MM.YYYY |
| End Date Field | Look for date input with label "Date de fin" | Format: DD.MM.YYYY |
| Country Dropdown | Look for select with label "Pays" | Select "France" |
| Category Dropdown | Look for select with label "Catégorie" | Select "Non refacturable" |
| Reason Field | Look for text input with label "Motif" | Text field |
| Comments Field | Look for textarea with label "Commentaires" | Text area |
| Submit Button | Look for button with text "Interrompre" | Submits the form |

## Important Notes

### Date Format
- Dates must be entered in DD.MM.YYYY format
- Example: April 17, 2026 = "17.04.2026"

### Iframe Access
- The form is loaded inside an iframe with ID `application-Expense-claim-iframe`
- All form interactions must be done through the iframe's contentDocument
- Example: `document.querySelector('#application-Expense-claim-iframe').contentDocument`

### Field Identification
- SAP WebDynpro forms use dynamic IDs (WD5B, WD6C, etc.)
- Use label text or placeholder attributes to identify fields
- Look for parent containers with class names like `lsControl`, `lsLabel`, etc.

### Dropdown Selection
- Dropdowns are **custom SAP WebDynpro components** — NOT standard HTML `<select>` elements
- **Use the `sap-dropdown` skill** for all dropdown interactions (Country, Category, Cat. frais, etc.)
- Never use `browser_click` with a simple CSS selector on a dropdown — it won't work
- Always use `browser_evaluate` with JavaScript injected into the iframe

## Example Workflow

```python
# Assuming user is already on the Expense Claims page
browser_navigate(url="https://nextgen.myatos.net/sap/flp#Expense-claim")
browser_wait(load_state="networkidle", timeout_ms=5000)

# Click the Create New Expense Report button
browser_evaluate(expression="""
  const iframe = document.querySelector('#application-Expense-claim-iframe');
  const button = iframe.contentDocument.querySelector('#WD5B');
  button.click();
""")

browser_wait(timeout_ms=3000)

# Fill in the form fields
# Start Date: 17.04.2026
# End Date: 17.04.2026
# Country: France
# Category: Non refacturable
# Reason: déplacement hackathon
# Comments: Test pour le hackathon

# Submit the form
# Click the Interrompre button
```

## Error Handling

### Form Not Loading
- If the form doesn't appear after clicking the Create button:
  - Wait longer with `browser_wait` (increase timeout to 5000ms)
  - Check browser logs with `browser_get_logs`
  - Verify the iframe is accessible

### Field Not Found
- If a field cannot be located:
  - Use `browser_screenshot` to see the current form state
  - Check the iframe's contentDocument for the field
  - Look for alternative selectors or labels

### Dropdown Not Working
- SAP WebDynpro dropdowns are NOT standard HTML `<select>` elements
- **Refer to the `sap-dropdown` skill** for detailed instructions on interacting with custom SAP dropdowns
- You must: click to open → wait → list options → scroll if needed → click the target option
- All dropdown interactions must be done via `browser_evaluate` inside the iframe

## Notes

- This skill assumes the form is in French (Atos MyAtos French interface)
- Field labels and button text are in French
- The form may have additional fields depending on the expense type
- Some fields may be optional or conditional based on previous selections
