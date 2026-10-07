"""Local HTTP adapter: python -m laya_api --model-path C:/models/laya."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys

from laya_client import LocalLaya, token_budgets

MAX_BODY_BYTES = 1024 * 1024


def make_handler(model, default_budgets=None):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'ready': True})
            else:
                self.reply(404, {'error': 'Use POST /classify or GET /health.'})

        def do_POST(self):
            # Close after the request, including rejected bodies that were not read.
            self.close_connection = True
            if self.path != '/classify':
                self.reply(404, {'error': 'Use POST /classify.'})
                return
            if self.headers.get('Transfer-Encoding'):
                self.reply(400, {'error': 'Send a JSON body with Content-Length.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1:
                    raise ValueError('A JSON request body is required.')
                if length > MAX_BODY_BYTES:
                    self.reply(413, {'error': 'Request body exceeds 1 MiB.'})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('JSON body must be an object with message and options.')
                budgets = dict(default_budgets or {})
                for key in ('max_len', 'head_max_len'):
                    if key in payload:
                        budgets[key] = payload[key]
                budgets = token_budgets(**budgets)
                result = model.classify(payload.get('message'), payload.get('options'), **budgets)
            except (ValueError, TypeError, UnicodeError) as error:
                self.reply(400, {'error': str(error)})
                return
            except Exception:
                print('Laya inference failed.', file=sys.stderr)
                self.reply(500, {'error': 'Model inference failed. Check the server.'})
                return
            self.reply(200, result)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, help='restored checkpoint directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max-len', type=int, help='default total input token limit')
    parser.add_argument('--head-max-len', type=int, help='default question/options token budget')
    args = parser.parse_args(argv)
    try:
        budgets = token_budgets(args.max_len, args.head_max_len)
        model = LocalLaya(args.model_path, device=args.device)
        with ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(model, budgets)) as server:
            print(f'Laya ready at http://127.0.0.1:{server.server_port}/classify', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
