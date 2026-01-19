import os
import json
import urllib.request

def post_to_slack(text: str) -> None:
    url = os.environ["SLACK_WEBHOOK_URL"]
    payload = {"text": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()

if __name__ == "__main__":
    post_to_slack("✅ pnl-tracker：Slack 通知管線已連線成功（測試訊息）")
    print("sent")