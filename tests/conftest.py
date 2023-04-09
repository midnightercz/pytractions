from pytest import fixture

from unittest.mock import patch

@fixture
def fixture_isodate_now():
    with patch("pytraction.base.isodate_now") as mocked_isodate_now:
        mocked_isodate_now.side_effect = [
            "1990-01-01T00:00:00.00000Z",
            "1990-01-01T00:00:01.00000Z",
            "1990-01-01T00:00:02.00000Z",
            "1990-01-01T00:00:03.00000Z",
            "1990-01-01T00:00:04.00000Z",
            "1990-01-01T00:00:05.00000Z",
            "1990-01-01T00:00:06.00000Z",
            "1990-01-01T00:00:07.00000Z",
            "1990-01-01T00:00:08.00000Z",
            "1990-01-01T00:00:09.00000Z",
            "1990-01-01T00:00:10.00000Z",
        ]
        yield mocked_isodate_now
