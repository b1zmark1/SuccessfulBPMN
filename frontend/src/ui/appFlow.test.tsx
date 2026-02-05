import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../state/useJobLifecycle", () => ({
  useJobLifecycle: () => ({
    jobId: null,
    job: null,
    isCreating: false,
    isPolling: false,
    requestError: null,
    submitJob: vi.fn(async () => undefined),
    reset: vi.fn(),
  }),
}));

import { App } from "../App";

describe("App flow", () => {
  it("shows two scenarios then expands selected one", () => {
    render(<App />);

    const scenarioButtons = screen.getAllByRole("button").filter((btn) =>
      btn.className.includes("scenario-card__trigger"),
    );
    expect(scenarioButtons.length).toBe(2);

    fireEvent.click(scenarioButtons[0]);

    expect(screen.getByRole("button", { name: /назад/i })).toBeInTheDocument();
    expect(screen.getByText(/png\/jpg\/webp/i)).toBeInTheDocument();
  });
});
