import asyncio


async def test_create_job_and_fetch_status(app_client):
    create_response = await app_client.post(
        "/api/v1/jobs",
        json={"job_type": "text_to_diagram", "meta": {"prompt": "build flow"}},
    )
    assert create_response.status_code == 202

    job_id = create_response.json()["job_id"]

    # Allow dispatcher to deliver and update queued status.
    for _ in range(20):
        get_response = await app_client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 200
        body = get_response.json()
        if body["status"] == "queued":
            break
        await asyncio.sleep(0.1)

    assert body["job_id"] == job_id
    assert body["job_type"] == "text_to_diagram"
    assert body["status"] in {"pending", "queued"}
    assert body["created_at"] is not None


async def test_get_unknown_job_returns_404(app_client):
    response = await app_client.get("/api/v1/jobs/018f4b89-0f90-7a9b-9f39-9ce21f744001")
    assert response.status_code == 404
