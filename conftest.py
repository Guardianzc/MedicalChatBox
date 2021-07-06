import pytest
import re


def pytest_addoption(parser):
    parser.addoption("--track", default='1,2,3', )


@pytest.fixture(scope='session')
def track(request):
    track_value = request.config.option.track
    if track_value is None:
        pytest.skip()
    return re.split(r'[;,，\s]\s*', track_value)
