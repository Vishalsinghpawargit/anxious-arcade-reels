"""Answer the Arcade Studio phone app's Claude requests with Claude Code, on the Claude subscription.

The phone app has no Claude of its own. When it needs a decision (which fights to keep, where a
requested fight starts and ends, which caster line makes a Speech Reel), it commits a request to
bridge/requests/<id>.json. That push runs this script in GitHub Actions, where Claude Code is logged
in with the CLAUDE_CODE_OAUTH_TOKEN secret, so the request uses the subscription and no API credit.
The answer goes to bridge/results/<id>.json, which the phone reads and then deletes.

Request:  {"id", "system", "prompt" (text or content blocks with images), "schema",
           "model" (default "haiku"), "thinking" (default 0)}
Result:   {"id", "ok", "answer" (matches schema) | "error", "cost_usd", "duration_ms"}
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUESTS, RESULTS = ROOT / 'requests', ROOT / 'results'


def ask(request):
    """One Claude Code request in print mode, answered as JSON matching the request's schema.

    Same command as the PC pipeline (YT-Downloader ai_clips.ask_claude): no tools, settings,
    plugins or MCP servers are loaded, so only the request's own prompt is sent.
    """
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')}
    env['MAX_THINKING_TOKENS'] = str(request.get('thinking', 0))
    cmd = ['claude', '-p', '--model', request.get('model', 'haiku'), '--system-prompt', request['system'],
           '--tools', '', '--setting-sources', '', '--strict-mcp-config', '--no-session-persistence',
           '--disable-slash-commands', '--json-schema', json.dumps(request['schema'])]
    prompt = request['prompt']
    if isinstance(prompt, str):
        cmd += ['--output-format', 'json']
    else:
        cmd += ['--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose']
        prompt = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': prompt}}) + '\n'

    out = {}
    # Now and then the model ends without a valid structured answer, so that is asked once more.
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix='bridge-') as cwd:
            result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', cwd=cwd, env=env, timeout=900)
        try:
            try:
                events = [json.loads(result.stdout)]
            except ValueError:
                events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            out = next(e for e in reversed(events) if isinstance(e, dict) and e.get('type') == 'result')
        except (ValueError, StopIteration):
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError('Claude Code failed: ' + (detail[-1] if detail else f'exit code {result.returncode}'))
        if out.get('is_error'):
            if out.get('subtype') == 'error_max_structured_output_retries':
                continue
            raise RuntimeError(f'Claude Code could not answer: {out.get("result") or out.get("subtype")}')
        if out.get('structured_output') is None:
            try:
                out['structured_output'] = json.loads(out.get('result') or '')
            except ValueError:
                continue
        return out
    raise RuntimeError(f'Claude did not return a structured answer ({out.get("subtype")})')


def main():
    RESULTS.mkdir(exist_ok=True)
    for path in sorted(REQUESTS.glob('*.json')):
        request_id = path.stem
        try:
            request = json.loads(path.read_text(encoding='utf-8'))
            out = ask(request)
            result = {'id': request_id, 'ok': True, 'answer': out['structured_output'],
                      'cost_usd': out.get('total_cost_usd'), 'duration_ms': out.get('duration_ms')}
        except Exception as e:  # the phone shows the message; the run itself stays green
            result = {'id': request_id, 'ok': False, 'error': str(e)[:1000]}
        (RESULTS / f'{request_id}.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        path.unlink()
        print(request_id, 'ok' if result['ok'] else 'error: ' + result['error'], flush=True)


if __name__ == '__main__':
    main()
