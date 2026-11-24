"""Secret stores under HERMES_HOME are write-denied; control files stay writable (#110464).

``get_read_block_error`` refuses every credential store. The write side is deliberately
narrower — #45947 freed ``auth.json`` / ``config.yaml`` / ``webhook_subscriptions.json`` so
the user can ask to edit them — but the secret *material* it meant to keep blocked had
drifted: ``auth/google_oauth.json``, the plaintext Bitwarden cache, ``vault/`` and
``browser-profile/`` were writable through ``write_file`` / ``patch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import agent.file_safety as fs

SECRET_STORES = (
    "auth/google_oauth.json", "cache/bws_cache.json", "vault/vault.key", "browser-profile/Default/Cookies",
)
WRITABLE_CONTROL_FILES = ("auth.json", "config.yaml", "webhook_subscriptions.json")


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Profile HERMES_HOME plus a distinct global root, both patched."""
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, profile


def _touch(base: Path, rel: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_read_denied_secret_stores_are_write_denied_on_profile_and_root(hermes_layout):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in SECRET_STORES:
            path = _touch(base, rel)
            assert fs.get_read_block_error(str(path)), f"fixture drift: not read-denied: {path}"
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_control_files_and_lookalikes_outside_home_stay_writable(hermes_layout, tmp_path):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in WRITABLE_CONTROL_FILES:
            assert fs.is_write_denied(str(_touch(base, rel))) is False, f"#45947 regression: {rel}"
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "cache/bws_cache.json"))) is False
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "vault/vault.key"))) is False


class TestProfileHomeProcessHome:
    """With the process HOME pinned to ``{HERMES_HOME}/home`` (TERMINAL_HOME_MODE=profile,
    containers, spawned workers) the write guards must still cover every home a write can
    land in: the OS user's real home, the profile home and ``~name`` accounts."""

    @pytest.fixture()
    def profile_home_env(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        (profile / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HOME", str(profile / "home"))
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: profile.parent)
        return profile

    def test_every_home_is_guarded(self, profile_home_env):
        import pwd

        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        for rel in (".aws/credentials", ".ssh/id_ed25519", ".netrc", ".config/gh/hosts.yml"):
            assert fs.is_write_denied(str(real_home / rel)), rel
            assert fs.is_write_denied(str(profile_home_env / "home" / rel)), rel
        assert fs.is_write_denied("~/.aws/credentials")
        assert fs.is_write_denied("~root/.ssh/authorized_keys")
        # ``~/.ssh/config`` stays approval-gated (not hard-denied) on the real home too.
        assert fs.is_write_approval_required(str(real_home / ".ssh" / "config"))
        assert fs.is_write_denied(str(real_home / ".ssh" / "config")) is False

    def test_benign_paths_stay_writable(self, profile_home_env, tmp_path):
        import pwd

        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        for benign in (tmp_path / "scratch" / "notes.txt", real_home / "projects" / "notes.md"):
            assert fs.is_write_denied(str(benign)) is False, benign
            assert fs.is_write_approval_required(str(benign)) is False, benign
        # Unknown ``~user`` is not a guard home, but credential *shape* still
        # hard-denies (home-independent) — fail closed, not open.
        assert fs.is_write_denied("~nosuchuser-hopefully/.ssh/authorized_keys") is True
        assert fs.is_write_denied("~nosuchuser-hopefully/projects/notes.md") is False


class TestRemoteAbsoluteCredentialShapes:
    """SSH (and other remote) backends expand ``~`` against a remote home that is
    not among Hermes-host ``_guard_homes``. Absolute remote credential paths must
    still hard-deny by shape; ``*/.ssh/config`` stays approval-gated."""

    REMOTE_HOME = "/home/deploy"

    def test_remote_ssh_keys_and_dotfiles_hard_denied(self):
        for rel in (
            ".ssh/authorized_keys",
            ".ssh/id_rsa",
            ".ssh/id_ed25519",
            ".ssh/known_hosts",
            ".ssh/some_key",
            ".netrc",
            ".pgpass",
            ".npmrc",
            ".pypirc",
            ".git-credentials",
        ):
            path = f"{self.REMOTE_HOME}/{rel}"
            assert fs.is_write_denied(path) is True, path
            assert fs.is_write_approval_required(path) is False, path
            err = fs.get_write_denied_error(path)
            assert err is not None and "protected system/credential file" in err, path

    def test_remote_ssh_config_approval_gated_not_hard_denied(self):
        path = f"{self.REMOTE_HOME}/.ssh/config"
        assert fs.is_write_denied(path) is False
        assert fs.is_write_approval_required(path) is True
        assert fs.get_write_denied_error(path) is None

    def test_shape_deny_independent_of_detected_remote_home(self, monkeypatch):
        """Absolute remote credential paths deny even with no live remote home."""
        monkeypatch.setattr(fs, "_detected_remote_homes", lambda: set())
        assert fs.is_write_denied(f"{self.REMOTE_HOME}/.ssh/authorized_keys") is True
        assert fs.is_write_approval_required(f"{self.REMOTE_HOME}/.ssh/config") is True

    def test_detected_remote_home_extends_prefix_denies(self, monkeypatch):
        """When SSH has confirmed ``$HOME``, non-shape prefixes (``.aws``) join the guards."""
        monkeypatch.setattr(fs, "_detected_remote_homes", lambda: {self.REMOTE_HOME})
        assert fs.is_write_denied(f"{self.REMOTE_HOME}/.aws/credentials") is True
        assert fs.is_write_denied(f"{self.REMOTE_HOME}/projects/notes.md") is False

    def test_host_home_denials_unchanged(self):
        assert fs.is_write_denied("~/.ssh/authorized_keys") is True
        assert fs.is_write_denied(os.path.expanduser("~/.ssh/id_ed25519")) is True
        assert fs.is_write_approval_required(os.path.expanduser("~/.ssh/config")) is True
        assert fs.is_write_denied(os.path.expanduser("~/.ssh/config")) is False
