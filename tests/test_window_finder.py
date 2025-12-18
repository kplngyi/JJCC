import pytest

from jj_chess_bot.capture import window_finder


def test_find_windows_by_name(monkeypatch):
    mock_windows = [
        {
            "kCGWindowName": "JJ象棋 - 游戏",
            "kCGWindowOwnerName": "JJ象棋Client",
            "kCGWindowNumber": 123,
            "kCGWindowOwnerPID": 456,
        },
        {
            "kCGWindowName": "Other",
            "kCGWindowOwnerName": "OtherApp",
            "kCGWindowNumber": 124,
            "kCGWindowOwnerPID": 789,
        },
    ]

    monkeypatch.setattr(window_finder, "list_windows", lambda: mock_windows)

    res = window_finder.find_windows_by_name("jj象棋")
    assert len(res) == 1
    assert res[0]["window_id"] == 123
    assert res[0]["pid"] == 456
    assert "JJ象棋" in res[0]["title"]
