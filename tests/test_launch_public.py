"""Small tests for public tunnel log parsing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from launch_public import extract_public_url


class PublicLauncherTests(unittest.TestCase):
    def test_extracts_latest_quick_tunnel_url(self):
        with TemporaryDirectory() as temporary:
            first = Path(temporary) / "stdout.log"
            second = Path(temporary) / "stderr.log"
            first.write_text("old https://first-field.trycloudflare.com\n", encoding="utf-8")
            second.write_text("ready https://New-Board-42.trycloudflare.com\n", encoding="utf-8")
            self.assertEqual(
                extract_public_url(first, second),
                "https://new-board-42.trycloudflare.com",
            )

    def test_extracts_localtunnel_url(self):
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "tunnel.log"
            log.write_text("https://Sophon-Board-7.loca.lt\n", encoding="utf-8")
            self.assertEqual(
                extract_public_url(log),
                "https://sophon-board-7.loca.lt",
            )

    def test_rejects_lookalike_domains(self):
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "tunnel.log"
            log.write_text(
                "https://trycloudflare.com.evil.example\n"
                "https://board.loca.lt.evil.example\n",
                encoding="utf-8",
            )
            self.assertIsNone(extract_public_url(log))


if __name__ == "__main__":
    unittest.main(verbosity=2)
