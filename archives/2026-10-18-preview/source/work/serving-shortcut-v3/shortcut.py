"""Opt-in double-click front end for the unchanged reviewed candidate launcher.

Only the delegated launcher may contact or start the server. Importing this
module and --validate-only are stdlib/CPU only. No baseline/profile writes,
process termination, inferred qualification, or automatic fallback occurs.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'outputs/flash1017-lab/qualified-candidate-profile.json'
MANIFEST = 'work/serving-profile-v3/staged-01/candidate-manifest.json'
LAUNCHER = 'work/serving-launch-v3/launch.py'
BOOTSTRAP_SHA = {
    'outputs/flash1017-lab/run.sh': 'fec7b7543f8d743f65ee321a6f3be7f583384941369cbaef9a1f129f801b9d27',
    LAUNCHER: 'f94c2959cc454af6403f91f6713c9f91e8aaf766b105eb98e3b8a69eff8e0674',
    'work/serving-launch-v3/manifest-01.json': '6060581b5cc68a36f6581df9e610ff11dc38dadf1ead0313477d04a9ea54041d',
    MANIFEST: '98c0532d46416e02698dc6a77476ea8152de7910cbd4641810c37e3195539032',
}


class Refused(ValueError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def save(path, data):
    pending = Path(path).with_suffix('.pending')
    with pending.open('w') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def require(value, message):
    if not value:
        raise Refused(message)


def validate(root=ROOT, *, pins=None):
    """Read-only gate; final HTTP qualification must have been granted by root."""
    root = Path(root).resolve()
    pins = BOOTSTRAP_SHA if pins is None else pins
    for relative, digest in pins.items():
        require(sha(root / relative) == digest, 'A reviewed launcher or manifest has changed. Startup was refused.')
    profile_path, manifest_path = root / PROFILE, root / MANIFEST
    require(profile_path.is_file(), 'The opt-in profile is not installed yet. Its HTTP checks must finish before this shortcut can start it.')
    profile, manifest = read(profile_path), read(manifest_path)
    require(profile.get('qualified') is True and profile.get('qualification_http_status') == 'complete_validated',
            'The opt-in profile has not completed HTTP qualification. Startup was refused.')
    require(profile.get('profile_kind') == 'flash1017-serving-candidate-v3', 'This is not the reviewed opt-in profile.')
    baseline_path = root / 'outputs/flash1017-lab/qualified-profile.json'
    baseline = read(baseline_path)
    require(sha(baseline_path) == manifest['baseline']['profile_sha256'], 'The native baseline changed. Startup was refused.')
    marker = {'name': 'cone-hotrows-v3', 'schema': 1, 'row_cache_bytes_per_table': 256 * 2**20,
              'manifest_sha256': sha(manifest_path)}
    require(profile['runtime'].get('serving_candidate') == marker, 'The opt-in profile does not match its reviewed manifest.')
    require({k: v for k, v in profile['runtime'].items() if k != 'serving_candidate'} == baseline['runtime'] and
            profile['packages'] == baseline['packages'] and profile['inference_files_sha256'] == baseline['inference_files_sha256'],
            'The opt-in profile changed the native settings or source pins.')
    require(type(profile.get('max_context_window')) is int and profile['max_context_window'] == manifest['max_context_window'] == 131072,
            'This shortcut is qualified only for the reviewed 131K context limit.')
    launcher_manifest = read(root / 'work/serving-launch-v3/manifest-01.json')
    for source, digest in (manifest['sources_sha256'] | launcher_manifest['sources_sha256']).items():
        require(sha(source) == digest, 'A pinned source or dependency changed. Startup was refused.')
    lab = root / 'outputs/flash1017-lab'
    for relative, digest in baseline['inference_files_sha256'].items():
        require(sha(lab / relative) == digest, 'A pinned native source changed. Startup was refused.')
    return {'profile': str(profile_path), 'profile_sha256': sha(profile_path), 'manifest': str(manifest_path),
            'manifest_sha256': sha(manifest_path), 'launcher_sha256': sha(root / LAUNCHER),
            'qualification_http_status': profile['qualification_http_status'], 'max_context_window': 131072,
            'namespace': hashlib.sha256(json.dumps(profile['runtime'], sort_keys=True).encode()).hexdigest()[:12]}


@contextmanager
def startup_lock(path):
    """Serialize shortcut attempts only; the frozen server owns the model lock."""
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused('Another Flash1017 startup check is already running. Wait for its window to finish; this shortcut did not start another model.') from exc
        yield


def failure_message(report):
    error = str(report.get('error', ''))
    if 'Another lab profile is running' in error:
        return 'Flash1017 is already running with another profile. Use Stop Flash1017, wait for it to finish, then open this shortcut again. No second model was started.'
    if not report.get('child_pid') and ('Address already in use' in error or 'Errno 48' in error):
        return 'The Flash1017 address is already in use. Stop the existing server before opening this shortcut again. No second model was started.'
    if report.get('child_pid') or 'TimeoutError' in error:
        return 'Startup could not be confirmed. The server may still be loading. Use Stop Flash1017 and wait for it to finish before starting a different profile.'
    return 'Flash1017 could not be confirmed ready. The saved launch report and log contain the details; this shortcut will not retry or switch profiles.'


def invoke(*, root=ROOT, validate_only=False, no_browser=False, runner=None, pins=None):
    """Return(code, durable invocation directory); no subprocess in validation mode."""
    root = Path(root).resolve()
    log_root = root / 'outputs/flash1017-lab/server-state/candidate-launches'
    log_root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + f'-{os.getpid()}-'
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=log_root))
    report_path = folder / 'shortcut-report.json'
    report = {'status': 'validating', 'started_at_unix': time.time(), 'invocation': str(folder),
              'shortcut_sha256': sha(__file__), 'launcher_invoked': False, 'validate_only': validate_only}
    save(report_path, report)
    code = 2
    try:
        with startup_lock(log_root / 'startup.lock'):
            checked = validate(root, pins=pins)
            report['validated'] = checked
            if validate_only:
                report.update(status='validation_passed', message='Source and profile checks passed. No server was contacted or started.')
                code = 0
            else:
                command = [str(root / 'outputs/flash1017-lab/run.sh'), '-u', str(root / LAUNCHER),
                           '--profile', checked['profile'], '--manifest', checked['manifest'],
                           '--log', str(folder / 'server.log'), '--report', str(folder / 'launch-report.json')]
                if no_browser:
                    command.append('--no-browser')
                report.update(status='starting', launcher_invoked=True, command=command)
                save(report_path, report)
                print('Checking and opening Flash1017 Fast. Loading can take a few minutes.', flush=True)
                with (folder / 'launcher.log').open('x') as log:
                    result = (subprocess.run if runner is None else runner)(command, cwd=root, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, check=False)
                report['launcher_exit_code'] = result.returncode
                launch_path = folder / 'launch-report.json'
                launched = read(launch_path) if launch_path.exists() else {}
                report['launch_report_sha256'] = sha(launch_path) if launch_path.exists() else None
                require(result.returncode == 0 and launched.get('status') == 'ready', failure_message(launched))
                require(all(launched.get(k) == checked[k] for k in ('profile', 'profile_sha256', 'manifest', 'manifest_sha256', 'namespace', 'launcher_sha256')),
                        'The ready server identity differs from the validated profile. Inspect the saved reports before starting anything else.')
                require(validate(root, pins=pins) == checked, 'Files changed during startup. Inspect the saved reports; do not start another profile until this server is stopped.')
                report.update(status='ready', message='Flash1017 Fast is ready. The native shortcut and default profile are unchanged.')
                code = 0
    except KeyboardInterrupt:
        report.update(status='interrupted', message='Startup was interrupted. A server may still be loading; use Stop Flash1017 and wait for it to finish before starting another profile.')
        code = 130
    except BaseException as exc:
        report.update(status='refused' if not report['launcher_invoked'] else 'failed', error=repr(exc),
                      message=str(exc) if isinstance(exc, Refused) else 'Flash1017 could not start. Details are in the saved shortcut report; no retry was attempted.')
    finally:
        report['completed_at_unix'] = time.time()
        save(report_path, report)
    print(report['message'], flush=True)
    print('Saved details: ' + str(folder), flush=True)
    return code, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    code, _ = invoke(validate_only=args.validate_only, no_browser=args.no_browser)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
