---
name: google-forms
description: Fill out Google Forms — text fields, radio buttons, checkboxes, dropdowns, and submit
browser_backend: direct
---

# Google Forms

## Site URL pattern
https://docs.google.com/forms/

## Important: Text Input Strategy

Google Forms uses custom Material Design components. The standard `browser_type` tool may **fail silently** on these fields because `fill("")` does not work on Google Forms inputs.

**Use `browser_evaluate` to type into text fields instead of `browser_type`:**

```js
// For short-answer fields (input)
const inputs = document.querySelectorAll('input[type="text"][aria-labelledby]');
const target = inputs[INDEX];  // INDEX = 0 for first text field, 1 for second, etc.
target.focus();
target.value = '';
const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
nativeInputValueSetter.call(target, 'YOUR ANSWER');
target.dispatchEvent(new Event('input', { bubbles: true }));
target.dispatchEvent(new Event('change', { bubbles: true }));
```

```js
// For paragraph/long-answer fields (textarea)
const textareas = document.querySelectorAll('textarea[aria-labelledby]');
const target = textareas[INDEX];
target.focus();
const nativeTextareaSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
nativeTextareaSetter.call(target, 'YOUR ANSWER');
target.dispatchEvent(new Event('input', { bubbles: true }));
target.dispatchEvent(new Event('change', { bubbles: true }));
```

## Workflow

1. Navigate to the form URL with `browser_navigate`
2. Use `browser_read_page` to read ALL questions at once
3. Answer questions **one by one from top to bottom** using the methods below
4. After all questions are answered, submit the form
5. Use `browser_read_page` to confirm submission

## Field Types and Selectors

### Short answer (text input)
- Selector: `input[type="text"][aria-labelledby]`
- **Use `browser_evaluate`** with the JS snippet above (not `browser_type`)
- Find which field by index: first text question = index 0, second = index 1, etc.

### Paragraph (long text)
- Selector: `textarea[aria-labelledby]`
- **Use `browser_evaluate`** with the textarea JS snippet above

### Radio buttons (single choice)
- Selector: `div[role="radio"][data-value="ANSWER_TEXT"]`
- Or: `browser_click` on `div[role="radio"]` that contains the answer text
- Alternative: `browser_click` on `span` or `label` containing the answer text within the question group
- Use: `browser_click` with selector `div[role="radio"][data-value="Option text"]`
- If `data-value` is not available, use: `browser_evaluate` with:
  ```js
  document.querySelectorAll('div[role="radio"]').forEach(r => {
    if (r.getAttribute('data-value') === 'YOUR ANSWER') r.click();
  });
  ```

### Checkboxes (multiple choice — can select several)
- Selector: `div[role="checkbox"][data-answer-value="ANSWER_TEXT"]`
- Or click the label/span containing the answer text
- Use: `browser_click` with selector `div[role="checkbox"][data-answer-value="Option text"]`
- If `data-answer-value` is not available, use `browser_evaluate`:
  ```js
  document.querySelectorAll('div[role="checkbox"]').forEach(c => {
    if (c.getAttribute('data-answer-value') === 'YOUR ANSWER') c.click();
  });
  ```
- **Check multiple boxes** by calling click for each option that applies

### Dropdown (select)
- Click the dropdown: `browser_click` on `div[role="listbox"]`
- Then click the option: `browser_click` on `div[role="option"][data-value="ANSWER_TEXT"]`
- Or use `browser_click` on `div[role="option"]` containing the text

### Date fields
- Selector: `input[type="date"]` or `input[aria-label="Day"]`, `input[aria-label="Month"]`, `input[aria-label="Year"]`
- Use `browser_evaluate` to set the value

## Discovering Questions and Answer Choices

Before answering, use `browser_evaluate` to list all questions and their options:

```js
// List all questions with their types and options
const questions = [];
document.querySelectorAll('[data-params]').forEach((q, i) => {
  const title = q.querySelector('[role="heading"]')?.textContent?.trim();
  const radios = [...q.querySelectorAll('div[role="radio"]')].map(r => r.getAttribute('data-value'));
  const checks = [...q.querySelectorAll('div[role="checkbox"]')].map(c => c.getAttribute('data-answer-value'));
  const textInput = q.querySelector('input[type="text"][aria-labelledby]');
  const textarea = q.querySelector('textarea[aria-labelledby]');
  let type = 'unknown';
  if (radios.length) type = 'radio';
  else if (checks.length) type = 'checkbox';
  else if (textInput) type = 'short_answer';
  else if (textarea) type = 'paragraph';
  questions.push({ index: i, title, type, options: radios.length ? radios : checks });
});
JSON.stringify(questions, null, 2);
```

## Submitting the Form

Click the submit button:
- `browser_click` on `div[role="button"]:has-text("Envoyer")` (French)
- Or: `div[role="button"]:has-text("Submit")` (English)
- Or use: `browser_evaluate` with:
  ```js
  document.querySelectorAll('div[role="button"]').forEach(b => {
    if (b.textContent.includes('Envoyer') || b.textContent.includes('Submit')) b.click();
  });
  ```

## Important Notes

- **Do NOT use `browser_type` for text fields** — it often fails on Google Forms. Use `browser_evaluate` instead.
- After using `browser_evaluate` to fill a field, the value may not visually appear immediately — that's OK, Google Forms registers it internally.
- Read ALL questions first, then answer them in order — don't re-read the page between each answer.
- For checkboxes, you can (and should) select multiple options when the question allows it.
- If a question is required and you skip it, the form will show an error on submit — check with `browser_read_page` after submitting.
