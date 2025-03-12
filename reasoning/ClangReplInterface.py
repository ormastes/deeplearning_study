from clang_repl_kernel import ClangReplKernel
from clang_repl_kernel.test_kernel import kernel


class ClangReplInterface:
    def __init__(self):
        ClangReplKernel.interactive = True
        self.kernel = ClangReplKernel()
        self.shell = self.kernel.my_shell

    def do_execute_sync(self, code):
        if code.strip().startswith('%<<'):
            code = code.strip()[3:]
            code = code[:-1] if code.endswith(';') else code
            code = "std::cout << " + code + " << std::endl;"
        # self.execution_count += 1
        try:
            output = self._do_execute_sync(code)
        except Exception as e:
            output = str(e)
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
                'output': output
            }

        return {
            'status': 'ok',
            # The base class increments the execution count
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
            'output': output
        }

    def _do_execute_sync(self, command):
        response_message = []

        def _send_response(msg):
            response_message.append(msg)

        self.shell.do_execute(command, _send_response)
        full_message = ''.join(response_message)
        return full_message