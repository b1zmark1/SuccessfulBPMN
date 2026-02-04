# UI Layer

## Screen Structure
1. Start screen:
   - Two equal scenario cards (`image_to_text`, `text_to_image`).
   - Selection expands chosen card to full workspace.
2. Input workspace:
   - Back button to return to start screen.
   - Scenario-specific form:
     - `image_to_text`: image URL input.
     - `text_to_image`: prompt textarea.
3. Status panel:
   - Shows `job_id`.
   - Shows backend status and polling state.
   - Handles unknown statuses safely.
4. Result panel:
   - `image_to_text`: extracted text (or JSON fallback).
   - `text_to_image`: preview + file download button.
   - `error`: backend error message.

## UX Flow
1. User selects a scenario on split start screen.
2. Selected scenario opens as a focused workspace.
3. User submits input; UI creates async job.
4. UI polls backend every ~2 seconds.
5. UI updates only from backend state.
6. Terminal states:
   - `done`: render result.
   - `error`: render error.
7. User can go back and start a new job.

## Design Direction
- Light theme with neutral base.
- Orange accent for key actions/status emphasis.
- Roboto typography.
- Desktop-first layout with responsive mobile fallback.
