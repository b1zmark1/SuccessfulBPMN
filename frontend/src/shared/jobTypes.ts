export const SUPPORTED_JOB_TYPES = ["image_to_text", "text_to_image"] as const;

export type SupportedJobType = (typeof SUPPORTED_JOB_TYPES)[number];

export type JobStatus = "pending" | "queued" | "running" | "done" | "error" | string;

export interface CreateJobRequest {
  job_type: SupportedJobType;
  meta: Record<string, unknown>;
}

export interface CreateJobResponse {
  job_id: string;
}

export interface JobResponse {
  job_id: string;
  job_type: string;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export const TERMINAL_STATUSES = new Set(["done", "error"]);
