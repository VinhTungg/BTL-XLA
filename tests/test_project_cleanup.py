"""Read-only checks for the opt-in Windows cleanup script. Never apply deletion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'clean_project.ps1'
POWERSHELL = shutil.which('powershell.exe')


@unittest.skipUnless(os.name == 'nt' and POWERSHELL, 'Windows PowerShell required')
class ProjectCleanupTest(unittest.TestCase):
    def test_default_mode_only_lists_scoped_disposable_targets(self):
        before = {p.name for p in ROOT.iterdir()}
        result = subprocess.run(
            [POWERSHELL, '-NoProfile', '-File', str(SCRIPT), '-Json'],
            cwd=ROOT, capture_output=True, check=True, timeout=60,
            encoding='utf-8', errors='replace',
        )
        report = json.loads(result.stdout)
        self.assertEqual(report['Mode'], 'dry_run')
        self.assertEqual(before, {p.name for p in ROOT.iterdir()})
        self.assertEqual(report['TotalBytes'], sum(t['Bytes'] for t in report['Targets']))
        allowed_roots = {'.venv-clean', '.pytest_cache', 'tmp/pdfs', 'tmp/uv-cache', '.git-initialized-backup'}
        git_files = {
            '.git/objects/8c/tmp_obj_jLGv0Z',
            '.git/objects/8c/tmp_obj_WHH6O0',
            '.git/objects/96/tmp_obj_RDE2Fw',
        }
        for target in report['Targets']:
            relative = Path(target['AbsolutePath']).relative_to(ROOT).as_posix()
            allowed = (
                relative in allowed_roots or relative in git_files or
                (relative.startswith(('src/', 'scripts/', 'tests/')) and
                 relative.endswith('/__pycache__'))
            )
            self.assertTrue(allowed, relative)
            self.assertEqual(target['Status'], 'planned')
            self.assertTrue(Path(target['AbsolutePath']).exists())
        for protected in ['.venv', 'data', 'artifacts', 'outputs', 'cifar100', 'docs']:
            self.assertIn(protected, report['Protected'])

    def test_script_parses_without_executing_cleanup(self):
        command = (
            '$tokens = $null; $parseErrors = $null; '
            '[System.Management.Automation.Language.Parser]::ParseFile('
            f"'{SCRIPT}', [ref]$tokens, [ref]$parseErrors) | Out-Null; "
            'if ($parseErrors.Count -gt 0) { $parseErrors | Out-String | Write-Error; exit 1 }; '
            "Write-Output 'CLEANUP_SYNTAX_OK'"
        )
        result = subprocess.run(
            [POWERSHELL, '-NoProfile', '-Command', command], cwd=ROOT,
            capture_output=True, check=True, timeout=30,
            encoding='utf-8', errors='replace',
        )
        self.assertIn('CLEANUP_SYNTAX_OK', result.stdout)
