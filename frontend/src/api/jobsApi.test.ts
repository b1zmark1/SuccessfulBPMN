import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "../test/setup";
import { createJob, getJob } from "./jobsApi";
import { ApiError } from "./errors";

const BASE = "http://localhost:8000/api/v1";

describe("jobsApi", () => {
  it("creates job with strict response parsing", async () => {
    server.use(
      http.post(`${BASE}/jobs`, async ({ request }) => {
        const body = (await request.json()) as { job_type: string; meta: Record<string, unknown> };
        expect(body.job_type).toBe("text_to_image");
        expect(body.meta.prompt).toBe("cat in city");
        expect(body.meta.promt).toBe("cat in city");
        return HttpResponse.json(
          { job_id: "12f23d89-270d-4f6e-bf30-fe5fbbf1f475" },
          { status: 202 },
        );
      }),
    );

    const result = await createJob({
      job_type: "text_to_image",
      meta: { prompt: "cat in city", promt: "cat in city" },
    });

    expect(result.job_id).toBe("12f23d89-270d-4f6e-bf30-fe5fbbf1f475");
  });

  it("maps backend error response to ApiError", async () => {
    server.use(
      http.get(`${BASE}/jobs/:id`, () =>
        HttpResponse.json({ detail: "Job not found" }, { status: 404 }),
      ),
    );

    await expect(getJob("018f4b89-0f90-7a9b-9f39-9ce21f744001")).rejects.toBeInstanceOf(ApiError);
  });
});
