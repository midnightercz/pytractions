from pytest import fixture

from unittest.mock import patch


@fixture
def fixture_isodate_now():
    with patch("pytractions.traction.isodate_now") as mocked_isodate_now:
        with patch("pytractions.tractor.isodate_now") as mocked_isodate_now2:
            with patch("pytractions.stmd.isodate_now") as mocked_isodate_now3:
                dates = [
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
                    "1990-01-01T00:00:11.00000Z",
                    "1990-01-01T00:00:12.00000Z",
                    "1990-01-01T00:00:13.00000Z",
                    "1990-01-01T00:00:14.00000Z",
                    "1990-01-01T00:00:15.00000Z",
                    "1990-01-01T00:00:16.00000Z",
                    "1990-01-01T00:00:17.00000Z",
                    "1990-01-01T00:00:18.00000Z",
                    "1990-01-01T00:00:19.00000Z",
                    "1990-01-01T00:00:21.00000Z",
                    "1990-01-01T00:00:22.00000Z",
                    "1990-01-01T00:00:23.00000Z",
                    "1990-01-01T00:00:24.00000Z",
                    "1990-01-01T00:00:26.00000Z",
                    "1990-01-01T00:00:27.00000Z",
                    "1990-01-01T00:00:28.00000Z",
                    "1990-01-01T00:00:29.00000Z",
                    "1990-01-01T00:00:30.00000Z",
                    "1990-01-01T00:00:31.00000Z",
                ]
                mocked_isodate_now.side_effect = dates
                mocked_isodate_now2.side_effect = dates
                mocked_isodate_now3.side_effect = dates

                yield mocked_isodate_now
