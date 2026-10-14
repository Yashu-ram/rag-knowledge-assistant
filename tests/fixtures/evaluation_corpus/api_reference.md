# KiwiDesk API Reference

The fictional KiwiDesk API base path is `/api/v2`.

The `POST /api/v2/tickets` endpoint creates a ticket and returns the identifier field `ticket_id`.

The `GET /api/v2/search` endpoint accepts the query parameter `q` and an optional `limit` parameter.

The default limit is 25.

API clients should send the `X-KiwiDesk-Workspace` header with each request.
