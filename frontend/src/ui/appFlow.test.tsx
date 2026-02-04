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

    expect(screen.getByRole("button", { name: /изображение в текст/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /текст в изображение/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /изображение в текст/i }));

    expect(screen.getByRole("button", { name: /назад/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/файл изображения/i)).toBeInTheDocument();
  });
});
