def pytest_runtest_setup(item):
    try:
        from pytest_socket import disable_socket

        disable_socket()
    except ImportError:
        pass
