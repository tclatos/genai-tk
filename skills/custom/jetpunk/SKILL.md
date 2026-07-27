---
name: jetpunk
description: Navigate JetPunk quiz site, take quizzes (typing, click, image, multiple choice), and handle cookie consent
browser_backend: direct
---

# JetPunk Quiz Site

## Site URL
https://www.jetpunk.com

## Cookie Consent

On first visit, a cookie consent banner may appear. Dismiss it before interacting with the quiz:
- Try clicking: `button` containing "Accepter" or "Accept" or "Consent"
- Or use: `browser_click` on `.fc-cta-consent` or `button.fc-cta-consent`
- If that fails, try: `browser_evaluate` with `document.querySelector('.fc-consent-root')?.remove()`

## Taking a Quiz

### Finding a quiz

- Random quiz: click on the "Random" link or navigate to `https://www.jetpunk.com/quizzes/random`
- Browse by category: navigate to `https://www.jetpunk.com/quizzes`
- Daily quiz: navigate to `https://www.jetpunk.com/daily-trivia`

### General workflow for ALL quiz types

1. Navigate to the quiz page
2. Dismiss cookie consent if present
3. Use `browser_read_page` to understand the quiz format and instructions
4. Use `browser_screenshot` to see the visual layout (especially for image quizzes)
5. Click the **Start Quiz** button if present: `button.start-quiz-btn` or element containing "Commencer" or "Start"
6. After EVERY action (click, type, etc.), ALWAYS call `browser_read_page` AND `browser_screenshot` to see the updated state
7. Continue answering until the quiz is complete (timer runs out or all answers given)

### Quiz types

#### Typing quizzes
1. The answer input field selector is: `input.txt-answer-box` or `input[id="txt-answer-box"]`
2. Type your answer into that input field, then press Enter
3. After each answer, use `browser_read_page` to see what changed

#### Click / Image quizzes (e.g. "par images click")
These quizzes show images and you must click on the correct one. **You MUST use `browser_screenshot` to see the images** — `browser_read_page` alone cannot show image content.

Workflow:
1. After starting the quiz, take a `browser_screenshot` to see all the images and the question
2. Read the question/prompt displayed on the page (use `browser_read_page`)
3. Look at the screenshot to identify which image matches the question
4. Click on the correct image using `browser_click` with the appropriate CSS selector (e.g. `img`, `.quiz-image`, or a specific clickable element) or coordinates
5. After clicking, IMMEDIATELY take another `browser_screenshot` and `browser_read_page` to see the next question
6. **REPEAT steps 2-5 in a loop** until the quiz ends — do NOT stop after one answer
7. Keep going until you see a results/score page or the timer runs out

Important for click quizzes:
- The clickable elements are usually `<img>` tags or `<div>` containers with images
- Use `browser_evaluate` with `document.querySelectorAll('.clickable-image, .quiz-image, .answer-image, img[data-answer]')` to discover the selectors
- If you cannot determine selectors from the DOM, use `browser_click` with **x,y coordinates** from the screenshot
- Some click quizzes highlight correct/wrong answers after clicking — check the screenshot to confirm

#### Multiple Choice Quizzes (Most Common)

This is the most common quiz type on JetPunk. The structure is:
- Question text displayed at the top
- Four answer options (A, B, C, D) displayed below
- Each answer is in a clickable element with class `.choice-button`

**How to answer:**
1. Read the question using `browser_read_page`
2. Identify the correct answer
3. Click on the answer using: `.choice-button:has-text("answer text")`
   - Example: `.choice-button:has-text("SARS")`
   - Or use: `.choice-button:has-text("A part of Europe")`
4. After clicking, the page will show:
   - Whether your answer was correct or incorrect
   - Percentage of other users who chose each option
   - Points earned (with time bonus)
   - A "Next question >" button to continue
5. Click "Next question >" to proceed to the next question
6. On the last question, the button will say "Finish quiz >" instead

**Important notes for multiple choice:**
- The `.choice-button` selector is the most reliable way to click answers
- You can use partial text matching with `:has-text()` - it doesn't need to be exact
- Always wait for the page to update after clicking before proceeding
- The quiz shows real-time statistics (percentage of users choosing each option)

#### Map quizzes
1. These involve clicking on a map — use `browser_click` with coordinates
2. Use `browser_screenshot` to see the map

## Important Notes

- Do NOT use the generic selector `input` — the page has many hidden inputs (search, ads, etc.)
- Always use **specific selectors** like `input.txt-answer-box` or `input[name="username"]`
- **NEVER stop after just one answer** — always continue the loop until the quiz is finished
- After EVERY interaction, call BOTH `browser_read_page` AND `browser_screenshot`
- If the page seems stuck or unchanged, try scrolling down or taking a screenshot to check for popups/overlays
- If typing fails, try `browser_evaluate` to focus the element first:
  ```js
  document.querySelector('input.txt-answer-box').focus()
  ```
- After typing an answer, press Enter by using `browser_evaluate`:
  ```js
  document.querySelector('input.txt-answer-box').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', bubbles: true}))
  ```

## Multiple Choice Quiz Details (Tested April 17, 2026)

### Answer Button Structure
- Each answer option is wrapped in a `.choice-holder` div
- Inside that is a `.choice` div
- The clickable element is `.choice-button` (can have class `active`)
- Inside the button is a `.choice-letter` span with the letter (A, B, C, D)

### Clicking Answers
The most reliable selector for clicking answers is:
```
.choice-button:has-text("answer text")
```

Examples that work:
- `.choice-button:has-text("SARS")`
- `.choice-button:has-text("The Tower of Babel")`
- `.choice-button:has-text("A part of Europe with no border control")`
- `.choice-button:has-text("Economist")`

You can use partial text - it doesn't need to match the entire answer text.

### Page Flow
1. Question displays with 4 answer options
2. Click an answer
3. Page updates to show:
   - "Correct!" or "Incorrect!" message
   - Points earned
   - Percentage breakdown of all users' choices
   - "Next question >" button (or "Finish quiz >" on last question)
4. Click "Next question >" to continue
5. Repeat until all questions answered

### Quiz Results
After finishing:
- Results page shows total score and percentage of users beaten
- All questions are listed with correct answers highlighted
- User statistics are displayed

## Session Management

After successfully completing a quiz, save the session:
```
browser_save_cookies(name="jetpunk")
```

To restore a previous session:
```
browser_load_cookies(name="jetpunk")
```

This preserves login state and quiz history.
