# ModMed Billing Automation

**Base URL:** `https://c5jampsaxdah62b3dco4ex4xgu0tzxxt.lambda-url.ap-south-1.on.aws`

---

## Option 1 — Swagger UI

Open in browser:

```
https://c5jampsaxdah62b3dco4ex4xgu0tzxxt.lambda-url.ap-south-1.on.aws/docs
```

### Step 1 — Upload a PDF

1. Click on **POST /upload** to expand it
2. Click **Try it out**
3. Under the `file` field, click **Choose File** and select your PDF fee ticket
4. Click **Execute**
5. Copy the `job_id` from the response

```json
{ "job_id": "a3f1c2d4e5b6...", "status": "queued" }
```

### Step 2 — Check Status

1. Click on **GET /status/{job_id}** to expand it
2. Click **Try it out**
3. Paste the `job_id` into the `job_id` field
4. Click **Execute**
5. Check the response:
   - `"status": "pending"` — still processing, wait a few seconds and Execute again
   - `"status": "success"` — done, results are in the response
   - `"status": "failed"` — something went wrong, check the `error` field

---

## Option 2 — curl

### Step 1 — Upload a PDF

```bash
curl -X POST https://c5jampsaxdah62b3dco4ex4xgu0tzxxt.lambda-url.ap-south-1.on.aws/upload \
  -F "file=@/path/to/fee_ticket.pdf"
```

Response:
```json
{ "job_id": "a3f1c2d4e5b6...", "status": "queued" }
```

### Step 2 — Check Status

```bash
curl https://c5jampsaxdah62b3dco4ex4xgu0tzxxt.lambda-url.ap-south-1.on.aws/status/<job_id>
```

- `"status": "pending"` — still processing, check again in a few seconds
- `"status": "success"` — done
- `"status": "failed"` — check the `error` field

### Watch Live Logs

Stream live Lambda logs to your terminal. Press **Ctrl+C** to stop.
Change `ap-south-1` to your region if different (default: `ap-south-1`).

```bash
trap 'pkill -f "aws logs tail" 2>/dev/null; exit' INT; stdbuf -oL aws logs tail /aws/lambda/mod-med-backend --follow --since 1m --region ap-south-1 | grep --line-buffered -v -E "mangum|No S3 records|App startup|App shutdown|START Request|END Request|REPORT Request" | sed -u 's/\[INFO\]/\x1b[32m[INFO]\x1b[0m/g; s/\[ERROR\]/\x1b[31m[ERROR]\x1b[0m/g; s/\[WARNING\]/\x1b[33m[WARNING]\x1b[0m/g'
```

Once you see the job complete in the logs, check the final result:

```bash
curl https://c5jampsaxdah62b3dco4ex4xgu0tzxxt.lambda-url.ap-south-1.on.aws/status/<paste-job-id-here>
```
