import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "../test/setup";
import { useJobLifecycle } from "./useJobLifecycle";

const BASE = "http://localhost:8000/api/v1";
const JOB_ID = "12f23d89-270d-4f6e-bf30-fe5fbbf1f475";

describe("useJobLifecycle", () => {
  it("polls until done", async () => {
    let pollCount = 0;

    server.use(
      http.post(`${BASE}/jobs`, () => HttpResponse.json({ job_id: JOB_ID }, { status: 202 })),
      http.get(`${BASE}/jobs/${JOB_ID}`, () => {
        pollCount += 1;
        const status = pollCount >= 2 ? "done" : "running";
        const result = status === "done" ? { image_url: "https://example.com/image.png" } : null;
        return HttpResponse.json(
          {
            job_id: JOB_ID,
            job_type: "text_to_image",
            status,
            result,
            error: null,
            created_at: "2026-02-04T10:00:00Z",
            started_at: "2026-02-04T10:00:01Z",
            finished_at: status === "done" ? "2026-02-04T10:00:05Z" : null,
          },
          { status: 200 },
        );
      }),
    );

    const { result } = renderHook(() => useJobLifecycle({ pollIntervalMs: 10 }));
    await act(async () => {
      await result.current.submitJob({
        jobType: "text_to_image",
        meta: { prompt: "river at sunset", promt: "river at sunset" },
      });
    });

    await waitFor(() => expect(result.current.job?.status).toBe("running"));
    await waitFor(() => expect(result.current.job?.status).toBe("done"));
    expect(result.current.isPolling).toBe(false);
  });

  it("stores error status from backend", async () => {
    server.use(
      http.post(`${BASE}/jobs`, () => HttpResponse.json({ job_id: JOB_ID }, { status: 202 })),
      http.get(`${BASE}/jobs/${JOB_ID}`, () =>
        HttpResponse.json(
          {
            job_id: JOB_ID,
            job_type: "image_to_text",
            status: "error",
            result: null,
            error: "worker failed",
            created_at: "2026-02-04T10:00:00Z",
            started_at: "2026-02-04T10:00:01Z",
            finished_at: "2026-02-04T10:00:05Z",
          },
          { status: 200 },
        ),
      ),
    );

    const { result } = renderHook(() => useJobLifecycle({ pollIntervalMs: 10 }));

    await act(async () => {
      await result.current.submitJob({ jobType: "image_to_text", meta: { image_url: "https://example.com/x.png" } });
    });

    await waitFor(() => expect(result.current.job?.status).toBe("error"));
    expect(result.current.job?.error).toBe("worker failed");
    expect(result.current.isPolling).toBe(false);
  });
});
