import { useCallback, useEffect, useRef, useState } from "react";
import { createJob, getJob } from "../api/jobsApi";
import { TERMINAL_STATUSES, type JobResponse, type SupportedJobType } from "../shared/jobTypes";

export interface JobSubmissionInput {
  jobType: SupportedJobType;
  meta: Record<string, unknown>;
}

export interface JobLifecycleState {
  jobId: string | null;
  job: JobResponse | null;
  isCreating: boolean;
  isPolling: boolean;
  requestError: string | null;
  submitJob: (input: JobSubmissionInput) => Promise<void>;
  reset: () => void;
}

interface JobLifecycleDeps {
  pollIntervalMs?: number;
  createJobFn?: typeof createJob;
  getJobFn?: typeof getJob;
}

const DEFAULT_POLL_INTERVAL_MS = 2000;

function normalizeError(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return "Непредвиденная ошибка";
}

function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function useJobLifecycle(deps: JobLifecycleDeps = {}): JobLifecycleState {
  const pollIntervalMs = deps.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const createJobFn = deps.createJobFn ?? createJob;
  const getJobFn = deps.getJobFn ?? getJob;

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearPollTimer = useCallback(() => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    clearPollTimer();
    setJobId(null);
    setJob(null);
    setIsCreating(false);
    setIsPolling(false);
    setRequestError(null);
  }, [clearPollTimer]);

  const submitJob = useCallback(async (input: JobSubmissionInput) => {
    clearPollTimer();
    setIsCreating(true);
    setIsPolling(false);
    setRequestError(null);
    setJob(null);
    setJobId(null);

    try {
      const created = await createJobFn({
        job_type: input.jobType,
        meta: input.meta,
      });
      setJobId(created.job_id);
      setIsPolling(true);
    } catch (error) {
      setRequestError(normalizeError(error));
    } finally {
      setIsCreating(false);
    }
  }, [clearPollTimer, createJobFn]);

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const response = await getJobFn(jobId);
        if (cancelled) {
          return;
        }

        setJob(response);
        setRequestError(null);
        if (isTerminal(response.status)) {
          setIsPolling(false);
          return;
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setRequestError(normalizeError(error));
      }

      pollTimeoutRef.current = setTimeout(poll, pollIntervalMs);
    };

    poll();

    return () => {
      cancelled = true;
      clearPollTimer();
    };
  }, [jobId, clearPollTimer, getJobFn, pollIntervalMs]);

  return {
    jobId,
    job,
    isCreating,
    isPolling,
    requestError,
    submitJob,
    reset,
  };
}
