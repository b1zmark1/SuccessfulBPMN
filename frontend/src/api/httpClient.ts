import { ApiError, NetworkError } from "./errors";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const API_PREFIX = import.meta.env.VITE_API_PREFIX ?? "/api/v1";

interface RequestOptions extends RequestInit {
  expectedStatus?: number;
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { expectedStatus, ...requestInit } = options;
  const url = `${API_BASE_URL}${API_PREFIX}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...(requestInit.headers ?? {}),
      },
      ...requestInit,
    });
  } catch {
    throw new NetworkError();
  }

  const hasBody = response.status !== 204;
  const payload = hasBody ? await response.json().catch(() => undefined) : undefined;

  if (!response.ok || (expectedStatus !== undefined && response.status !== expectedStatus)) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof (payload as Record<string, unknown>).detail === "string"
        ? (payload as { detail: string }).detail
        : `Ошибка запроса, статус ${response.status}`;
    throw new ApiError(message, response.status, payload);
  }

  return payload as T;
}
