import pytest
from ClangReplInterface import ObjectPool, ClangReplInterface, ObjectPoolLog

# Define a fixture that returns common initialization variables.
@pytest.fixture
def clang_repl():
    clang_repl = ClangReplInterface()
    return clang_repl

def test_simle_eval(clang_repl):
    # Test the simple_eval method
    results = clang_repl.run_verify(">>>%<< 1 + 1;\n2")
    state = results[0]
    message = results[1] if len(results) > 1 else ""
    assert state == 'ok', f"Expected state 'ok' but got '{state}'. Additional info: {message}"

def test_init_exception(clang_repl):
    dummy_exception = Exception("Dummy exception")
    clang_repl.dump_status(dummy_exception)

def test_validate(clang_repl):
    results = clang_repl.validate("int add(int a, int b){return a+b;}",">>> %<< add(1,1);\n2")
    state = results[0]
    message = results[1] if len(results) > 1 else ""
    assert state == None, f"Expected state 'ok' but got '{state}'. Additional info: {message}"


@pytest.fixture
def pool():
    def reward(clang_repl, text):
        return clang_repl.run_verify(text)
    return ObjectPool(ClangReplInterface, reward, 4)

def test_object_pool_run_verify(pool):
    try:
        code_snippet = ">>>%<< 1 + 1;\n2"
        pool.start_tasks([[code_snippet]]*4)
        results = pool.get_results(timeout=15)
        state = results[0]
        message = results[1] if len(results) > 1 else ""
        assert all(status == 'ok' for status in state), (
            f"Expected all statuses to be 'ok' but got {state}. Additional info: {message}"
        )
    finally:
        pool.stop()

def test_object_pool_run_verify_log(pool):
    try:
        pool.LOG=ObjectPoolLog.INFO
        code_snippet = ">>>%<< 1 + 1;\n2"
        pool.start_tasks([[code_snippet]]*4)
        results = pool.get_results(timeout=15)
        state = results[0]
        message = results[1] if len(results) > 1 else ""
        assert all(status == 'ok' for status in state), (
            f"Expected all statuses to be 'ok' but got {state}. Additional info: {message}"
        )
    finally:
        pool.stop()




def test_object_pool_run_verify_log_timeout(pool):
    try:
        pool._get_results = lambda self, f, timeout: (_ for _ in ()).throw(TimeoutError("Timeout"))
        pool.LOG=ObjectPoolLog.INFO
        code_snippet = ">>>%<< 1 + 1;\n2"
        pool.start_tasks([[code_snippet]]*4)
        results = pool.get_results(timeout=15)
        state = results[0]
        message = results[1] if len(results) > 1 else ""
        assert all(status == 'ok' for status in state), (
            f"Expected all statuses to be 'ok' but got {state}. Additional info: {message}"
        )
    finally:
        pool.stop()


def test_object_pool_run_verify_multiple_times(pool):
    try:
        pool.LOG=ObjectPoolLog.INFO
        code_snippet = ">>>%<< 1 + 1;\n2"
        for i in range(100):
            pool.start_tasks([[code_snippet]]*4)
            results = pool.get_results(timeout=15)
            state = results[0]
            message = results[1] if len(results) > 1 else ""
            assert all(status == 'ok' for status in state), (
                f"Expected all statuses to be 'ok' but got {state}. Additional info: {message}"
            )
    finally:
        pool.stop()