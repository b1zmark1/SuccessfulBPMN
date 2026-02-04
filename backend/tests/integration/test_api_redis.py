import asyncio
import json


async def test_post_job_publishes_redis_stream(app_client, redis_client):
    await redis_client.delete("jobs:test:stream")

    response = await app_client.post(
        "/api/v1/jobs",
        json={"job_type": "image_to_table", "meta": {"image_url": "https://example.org/a.png"}},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(20):
        entries = await redis_client.xrange("jobs:test:stream", min="-", max="+", count=10)
        if entries:
            break
        await asyncio.sleep(0.1)

    assert entries, "Expected at least one Redis stream message"
    _, payload = entries[-1]
    assert payload["JobID"] == job_id
    assert payload["job_type"] == "image_to_table"
    assert json.loads(payload["Metadata"]) == {"image_url": "https://example.org/a.png"}
