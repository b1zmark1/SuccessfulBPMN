import { requestJson } from "./httpClient";
import type { CreateJobRequest, CreateJobResponse, JobResponse } from "../shared/jobTypes";

const UUID_REGEXP = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const UUID_EXTRACT_REGEXP = /[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/i;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function assertCreateJobResponse(value: unknown): asserts value is CreateJobResponse {
  if (!isObject(value) || typeof value.job_id !== "string" || !UUID_REGEXP.test(value.job_id)) {
    throw new TypeError("Invalid create job response shape");
  }
}

function assertJobResponse(value: unknown): asserts value is JobResponse {
  if (!isObject(value)) {
    throw new TypeError("Invalid job response: not an object");
  }

  const hasValidCoreFields =
    typeof value.job_id === "string" &&
    typeof value.job_type === "string" &&
    typeof value.status === "string" &&
    typeof value.created_at === "string";

  if (!hasValidCoreFields) {
    throw new TypeError("Invalid job response: missing required fields");
  }

  if (value.result !== null && value.result !== undefined && !isObject(value.result)) {
    throw new TypeError("Invalid job response: result must be object or null");
  }

  if (value.error !== null && value.error !== undefined && typeof value.error !== "string") {
    throw new TypeError("Invalid job response: error must be string or null");
  }
}

export async function createJob(payload: CreateJobRequest): Promise<CreateJobResponse> {
  const data = await requestJson<unknown>("/jobs", {
    method: "POST",
    expectedStatus: 202,
    body: JSON.stringify(payload),
  });
  assertCreateJobResponse(data);
  return data;
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const match = jobId.match(UUID_EXTRACT_REGEXP);
  if (!match) {
    throw new TypeError("Invalid job id format");
  }

  const normalizedJobId = match[0];
  const data = await requestJson<unknown>(`/jobs/${normalizedJobId}`, {
    method: "GET",
    expectedStatus: 200,
  });
  assertJobResponse(data);

  return {
    job_id: data.job_id,
    job_type: data.job_type,
    status: data.status,
    result: data.result ?? null,
    error: data.error ?? null,
    created_at: data.created_at,
    started_at: typeof data.started_at === "string" ? data.started_at : null,
    finished_at: typeof data.finished_at === "string" ? data.finished_at : null,
  };
}
