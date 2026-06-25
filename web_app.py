import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from spam_classifier.paths import DEFAULT_MODEL_PATH, PIPELINE_MODEL_DIR
from spam_classifier.pipeline import load_pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_EXAMPLES = [
    "Win a free iPhone now! Click this link to claim your prize",
    "Let's meet tomorrow at 8 for coffee",
    "URGENT! Your account has been selected for a cash reward. Reply YES now",
]


def newest_pipeline_path():
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    candidates = sorted(
        PIPELINE_MODEL_DIR.glob("*_pipeline.joblib"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else DEFAULT_MODEL_PATH


def render_page(result=None, message="", model_path=None, error=None):
    escaped_message = html.escape(message)
    model_text = html.escape(str(model_path or newest_pipeline_path()))
    result_html = ""
    if result:
        label_class = "spam" if result["label"] == "Spam" else "ham"
        confidence = result.get("confidence")
        confidence_text = f"<span>{confidence:.2%} confidence</span>" if confidence is not None else ""
        result_html = f"""
        <section class="result {label_class}">
          <p class="eyebrow">Prediction</p>
          <h2>{result["label"]}</h2>
          {confidence_text}
        </section>
        """

    error_html = f'<section class="error">{html.escape(error)}</section>' if error else ""
    examples = "".join(
        f'<button type="button" data-example="{html.escape(example)}">{html.escape(example)}</button>'
        for example in DEFAULT_EXAMPLES
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SMS Spam Classifier</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5c677d;
      --line: #d8deea;
      --panel: #ffffff;
      --bg: #eef3f8;
      --accent: #176b87;
      --spam: #b42318;
      --spam-bg: #fff1f0;
      --ham: #087443;
      --ham-bg: #ecfdf3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      width: min(980px, calc(100% - 32px));
      margin: 32px auto;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1.15;
    }}
    .subtle {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    textarea {{
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: 16px/1.5 Arial, Helvetica, sans-serif;
    }}
    textarea:focus {{
      outline: 3px solid rgba(23, 107, 135, 0.18);
      border-color: var(--accent);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 700;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .examples {{
      display: grid;
      gap: 8px;
    }}
    .examples button {{
      text-align: left;
      font-weight: 400;
      line-height: 1.35;
    }}
    .result {{
      border-radius: 8px;
      padding: 18px;
      border: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    .result h2 {{
      margin: 0;
      font-size: 34px;
    }}
    .result span {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
    }}
    .result.spam {{
      color: var(--spam);
      background: var(--spam-bg);
      border-color: #f2b8b5;
    }}
    .result.ham {{
      color: var(--ham);
      background: var(--ham-bg);
      border-color: #abefc6;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .error {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
      color: #9a3412;
    }}
    code {{
      display: block;
      margin-top: 8px;
      padding: 10px;
      border-radius: 6px;
      background: #f6f8fb;
      color: #27364f;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 24px, 980px); margin-top: 20px; }}
      .workspace {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>SMS Spam Classifier</h1>
      <p class="subtle">Test spam/ham predictions from the refactored scikit-learn pipeline.</p>
    </header>
    <div class="workspace">
      <section class="panel">
        {error_html}
        {result_html}
        <form method="post" action="/predict">
          <label for="message">Message</label>
          <textarea id="message" name="message" required>{escaped_message}</textarea>
          <div class="actions">
            <button class="primary" type="submit">Predict</button>
            <button type="button" id="clear">Clear</button>
          </div>
        </form>
      </section>
      <aside class="panel">
        <p class="eyebrow">Model</p>
        <code>{model_text}</code>
        <p class="eyebrow" style="margin-top: 18px;">Examples</p>
        <div class="examples">{examples}</div>
      </aside>
    </div>
  </main>
  <script>
    const box = document.querySelector("#message");
    document.querySelectorAll("[data-example]").forEach((button) => {{
      button.addEventListener("click", () => {{
        box.value = button.dataset.example;
        box.focus();
      }});
    }});
    document.querySelector("#clear").addEventListener("click", () => {{
      box.value = "";
      box.focus();
    }});
  </script>
</body>
</html>"""


class SpamClassifierHandler(BaseHTTPRequestHandler):
    pipeline = None
    model_path = None

    def send_html(self, body, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.send_html(render_page(model_path=self.model_path))

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        message = parse_qs(raw_body).get("message", [""])[0].strip()

        if not message:
            self.send_html(render_page(message=message, model_path=self.model_path, error="Please enter a message."))
            return

        try:
            prediction = int(self.pipeline.predict([message])[0])
            label = "Spam" if prediction == 1 else "Ham"
            result = {"label": label}
            model = self.pipeline.named_steps["model"]
            if hasattr(model, "predict_proba"):
                result["confidence"] = float(self.pipeline.predict_proba([message])[0][prediction])
            self.send_html(render_page(result=result, message=message, model_path=self.model_path))
        except Exception as exc:
            self.send_html(
                render_page(message=message, model_path=self.model_path, error=f"Prediction failed: {exc}"),
                status=500,
            )

    def log_message(self, format, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small web UI for the spam classifier.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path) if args.model_path else newest_pipeline_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train one first, for example: python train.py --model lr"
        )

    SpamClassifierHandler.model_path = model_path
    SpamClassifierHandler.pipeline = load_pipeline(model_path)

    server = ThreadingHTTPServer((args.host, args.port), SpamClassifierHandler)
    print(f"Frontend running at http://{args.host}:{args.port}")
    print(f"Using model: {model_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
