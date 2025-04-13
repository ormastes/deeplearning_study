from clang_repl_kernel import ClangReplKernel, update_platform_system, is_installed_clang_exist, install_bundles, ClangReplConfig
import platform
import os
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import pickle
import subprocess
import tempfile
import os
import re
import enum
import string
from pathlib import Path

class Validator:
    def __init__(self, clang_repl, lib_path='/x86_64-linux-gnu/lib'):
        self.program = clang_repl.shell.program.replace("clang-repl", "clang++")
        self.env = clang_repl.shell.env.copy()

    def validate(self, codes):
        result = self.check_cpp_basic_charset(codes)
        if not result[0]:
            return (0.5, result[1])
        result = self.check_preprocess_and_tokenize(codes)
        if not result[0]:
            return (0.7, result[1])
        result = self.check_source_compilable(codes)
        if not result[0]:
            return (0.9, result[1])
        return (None, None)


    def check_cpp_basic_charset(self, s: str):
        """
        Check if every character in the string `s` is part of the C++ basic source character set.
        
        This set includes:
        - All letters (A–Z, a–z)
        - All digits (0–9)
        - Punctuation: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
        - Whitespace: space, \t, \n, \r, \f, \v
        """
        allowed = set(string.printable)
        for idx, c in enumerate(s):
            if c not in allowed:
                return (False, f'Char "{c}" at {idx} not in C++ basic source character set')
        return (True, None)

    def check_source_compilable(self, source: str):
        # Create a temporary file for the source code
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w", encoding="utf-8") as src_file:
            src_file.write(source)
            src_file_path = src_file.name

        # Prepare a temporary filename for the output executable
        output_path = src_file_path + ".o"

        try:
            # Invoke clang++ to compile the source file.
            # You may adjust the command-line options as needed.
            result = subprocess.run(
                [self.program, src_file_path,  "-c",  "-o", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env
            )

            if result.returncode == 0:
                return (True, None)
            else:
                #print("Compilation failed with errors:")
                #print(result.stderr)
                return (False, result.stderr.strip())
        finally:
            # Clean up the temporary files
            os.remove(src_file_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    def check_preprocess_and_tokenize(self, source: str):
        # Write source code to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w", encoding="utf-8") as src_file:
            src_file.write(source)
            src_file_path = src_file.name

        try:
            # Run clang++ with flags to perform syntax checking and dump tokens.
            result = subprocess.run(
                [self.program, "-fsyntax-only", "-Xclang", "-dump-tokens", src_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env
            )
            
            # If compilation succeeds, return True.
            if result.returncode == 0:
                return (True, None)
            else:
                # Try to parse the error message for line and column information.
                # Clang error messages typically look like:
                # "filename:line:col: error: <message>"
                pattern = r"^(.*):(?P<line>\d+):(?P<col>\d+):\s+(?:error|fatal error):\s+(?P<msg>.*)$"
                match = re.search(pattern, result.stderr, re.MULTILINE)
                if match:
                    error_info = {
                        "line": int(match.group("line")),
                        "col": int(match.group("col")),
                        "message": match.group("msg").strip()
                    }
                else:
                    error_info = {"message": result.stderr.strip()}
                return (False, error_info)
        finally:
            # Clean up temporary file.
            os.remove(src_file_path)

class ObjectPoolLog(enum.Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2

class ObjectPool:
    LOG=ObjectPoolLog.ERROR
    def __init__(self, cls, method, batch_size, *args, **kwargs):
        self.cls = cls                      # class to instantiate
        self.method = method               # unbound method like Worker.process
        self.args = args                   # args for cls(...)
        self.kwargs = kwargs               # kwargs for cls(...)
        self.batch_size = batch_size
        self.pool = []
        self.pool.append(self.cls(*self.args, **self.kwargs))
        self.lock = threading.Lock()
        self.running = True
        self.futures: list[Future] = []
        self.creation_futures =[]
        self.input_map = {}
        self.pool_max = self.batch_size*7
        self.pool_min = self.batch_size*7

        self.executor = ThreadPoolExecutor(max_workers=self.pool_max)
        self.creator_executor = ThreadPoolExecutor(max_workers=self.pool_min)
        self.thread = threading.Thread(target=self._maintain_pool, daemon=True)
        self.thread.start()
        while len(self.pool) < self.pool_min:
            time.sleep(0.05)

    def print_status(self, log, obj=None):
        if ObjectPool.LOG.value <= log.value:
            print("ObjectPool status:")
            print(f"ObjectPool status: {len(self.pool)} objects in pool, {len(self.futures)} futures in progress.")
            print(f"Creation futures: {len(self.creation_futures)}")
            print(f"Input map: {len(self.input_map)} entries")
            print(f"Pool max: {self.pool_max}, Pool min: {self.pool_min}")
            print(f"Executor max workers: {self.executor._max_workers}, Creator executor max workers: {self.creator_executor._max_workers}")
            print(f"Running: {self.running}")
            print(f"Thread alive: {self.thread.is_alive()}")
            if obj is not None:
                if isinstance(obj, ClangReplInterface):
                    obj.dump_status()
                else:
                    print(obj)

    def _maintain_pool(self):
        while self.running:
            with self.lock:
                need = self.pool_max - len(self.pool)
    
            if need > 0:
                self.creation_futures = []
                for _ in  range(need):
                    self.creation_futures.append(self.creator_executor.submit(self.cls, *self.args, **self.kwargs))
                    time.sleep(0.1)
                
                for f in self.creation_futures:
                    obj = f.result()
                    if obj.ok:
                        with self.lock:
                            self.pool.append(obj)


            self.creation_futures =[]
            time.sleep(0.1)

    def get_objects(self, n):
        while True:
            with self.lock:
                if len(self.pool) >= n:
                    return [self.pool.pop(0) for _ in range(n)]
            time.sleep(1)

    def start_tasks(self, args_list):
        self.print_status(ObjectPoolLog.INFO, "start_tasks started")
        while len(self.pool)+len(self.creation_futures) < self.pool_min:
            time.sleep(1)
        self.print_status(ObjectPoolLog.INFO, "start_tasks pool all running")
        objs = self.get_objects(len(args_list))
        self.print_status(ObjectPoolLog.INFO, "start_tasks got all objects")
        time.sleep(0.1)
        futures = []
        for obj, args in zip(objs, args_list):
            f = self.executor.submit(self.method, obj, *args)
            self.input_map[f] = (obj, args)
            futures.append(f)
            time.sleep(1)
        self.futures.extend(futures)
        time.sleep(0.1)
        self.print_status(ObjectPoolLog.INFO, "start_tasks done")

    def log_timeout(self, obj, args, log_header="", log_post=""):
        a_min_load, _, _ = os.getloadavg()
        progress = obj.get_progress()
        if progress is None:
            print("get progress is none:", obj)
            progress = 0.0
        score = 1.24 + progress * 0.8 if progress > 0.0 else 0
        error_msg = (f"{log_header}TimeoutError (ObjectPool.get_results): "
                     f"score={score:.2f}, CPU load={a_min_load:.2f}/32{log_post}")
        print(error_msg)
        return (score, error_msg)
    
    def _get_result(self, f, timeout):
        return f.result(timeout=timeout)

    def get_results(self, timeout=10.0):
        self.print_status(ObjectPoolLog.INFO, "get_results started")
        collected = [None] * len(self.futures)
        done_futures = []
        # Build a map of futures to their index positions.
        feature_idx_map = {f: idx for idx, f in enumerate(self.futures)}
                    
        # Wait for the system load to drop below the threshold.
        load_threshold = 4.0 + 1.0
        for i in range(int(timeout)):
            if os.getloadavg()[0] <= load_threshold:
                break
            time.sleep(1)
        self.print_status(ObjectPoolLog.INFO, "get_results system load check done")
        
        # --- Handle already finished futures first with a short timeout ---
        finished_futures = [f for f in self.futures if f.done()]
        for f in finished_futures:
            try:
                # Use a shorter timeout for finished futures (should normally return immediately).
                result = self._get_result(f, timeout=1)
            except TimeoutError:
                obj, args = self.input_map[f]
                result = self.log_timeout(obj, args, log_post=", input:" + str(args))
                self.print_status(ObjectPoolLog.ERROR, obj)

            except Exception as e:
                result = (0.0, f"Exception {e}")
                print("Exception:", e)
            finally:
                if not isinstance(result, (list, tuple)):
                    result = [result]
                # Use feature_idx_map to insert the result at the correct position.
                collected[feature_idx_map[f]] = result
                done_futures.append(f)
        self.print_status(ObjectPoolLog.INFO, f"get_results already done futures processed({len(done_futures)})")
        
        # Remove the finished futures from tracking.
        for f in done_futures:
            if f in self.futures:
                del self.input_map[f]
                self.futures.remove(f)
        
        # --- Process remaining futures with the standard timeout ---
        # Iterate over a copy of self.futures because we modify the list.
        for f in self.futures[:]:
            try:
                result = self._get_result(f, timeout=timeout)
            except TimeoutError:
                obj, args = self.input_map[f]
                result = self.log_timeout(obj, args, log_post=", input:" + str(args))
                self.print_status(ObjectPoolLog.ERROR, obj)
                # Retry once: re-submit the same task.
                new_obj = self.get_objects(1)[0]
                retry_future = self.executor.submit(self.method, new_obj, *args)
                try:
                    result = self._get_result(retry_future, timeout=timeout * 2)
                    print("Retry succeeded after retry")
                except TimeoutError:
                    result = self.log_timeout(new_obj, args, log_header="Retry failed: ")
                    self.print_status(ObjectPoolLog.ERROR, obj)
            except Exception as e:
                result = (0.0, f"Exception {e}")
                print("Exception:", e)
            finally:
                if not isinstance(result, (list, tuple)):
                    result = [result]
                collected[feature_idx_map[f]] = result
                done_futures.append(f)
        
        # Remove all processed futures.
        for f in done_futures:
            if f in self.futures:
                del self.input_map[f]
                self.futures.remove(f)
    
        # Transpose the collected results.
        transposed = list(map(list, zip(*collected)))
        self.print_status(ObjectPoolLog.INFO, "get_results done")
        return transposed if len(transposed) > 1 else transposed[0]


    def stop(self):
        self.running = False
        self.thread.join()
        self.executor.shutdown(wait=True)


class ClangReplInterface:
    SHELL_ERR_COUNT = 0
    def __init__(self):
        if not is_installed_clang_exist():
            install_bundles(platform.system())
        if not ClangReplKernel.ClangReplKernel_InTest:
            ClangReplKernel.ClangReplKernel_InTest = True
        self.line_count = sys.maxsize
        self.outputs = []
        self.processed = []
        self.results = []
        self.last_code = None
        try:
            self.kernel = None
            self.kernel = ClangReplKernel() 
            self.shell = None
            self.shell = self.kernel.my_shell
            self.kernel.my_shell.run()
            state, log = self.run_verify(">>> %<<0\n0")
            self.validator = Validator(self)
            if state != 'ok':
                raise Exception("ClangReplInterface: Kernel not ready, can not handle simple eval")
            self.execution_count = 0
            self.ok = True
        except Exception as e:
            self.dump_status(e)
            self.ok = False



    def dump_status(self, e=None):
        log_file = f'err_log/clang_err_{ClangReplInterface.SHELL_ERR_COUNT}.log'
        ClangReplInterface.SHELL_ERR_COUNT += 1

        os.makedirs('err_log', exist_ok=True)

        # Open the file for writing.
        with open(log_file, 'w') as f:
            # Define a helper function to print to both console and file.
            def dual_print(*args, **kwargs):
                print(*args, **kwargs)
                print(*args, **kwargs, file=f)

            try:
                dual_print("ClangReplInterface creation error:", e)
                dual_print("Last codes run:", self.last_code)
                dual_print("Processed codes:", self.processed)
                dual_print("Outputs:", self.outputs)

                if getattr(self, "kernel", None) is None:
                    dual_print("self.kernel is None")
                elif getattr(self.kernel, "my_shell", None) is None:
                    dual_print("self.kernel.my_shell is None")
                else:
                    if getattr(self.kernel.my_shell, "_prog", None) is None:
                        dual_print("self.kernel.my_shell._prog is None")
                    else:
                        dual_print("ClangReplInterface prog:", self.kernel.my_shell._prog)
                    if getattr(self.kernel.my_shell, "args", None) is None:
                        dual_print("self.kernel.my_shell.args is None")
                    else:
                        dual_print("ClangReplInterface args:", self.kernel.my_shell.args)
                    if getattr(self.kernel.my_shell, "env", None) is None:
                        dual_print("self.kernel.my_shell.env is None")
                    else:
                        dual_print("ClangReplInterface env:", self.kernel.my_shell.env)
            except Exception as e2:
                dual_print("ClangReplInterface creation error in exception block:", e2)


        
    def close(self):
        # Shut down the shell loop gracefully
        try:
            if hasattr(self.kernel.my_shell, 'del_loop'):
                self.kernel.my_shell.del_loop()
        except Exception as e:
            print("Error while stopping shell loop:", e)
        # Kill the subprocess
        try:
            if hasattr(self.kernel.my_shell, 'process') and self.kernel.my_shell.process is not None:
                self.kernel.my_shell.process.kill()
        except Exception as e:
            print("Error while killing process:", e)

    def __del__(self):
        # As a fallback, ensure resources are cleaned up when the object is deleted
        self.close()
        
    def get_progress(self):
        return len(self.outputs)/self.line_count

    def do_execute_sync(self, code):
        return self.kernel.do_execute_sync(code)
    
    def validate(self, org_target_text, clang_repl_test):
        cpp_code = ""
        for header in ClangReplConfig.HEADERS:
            cpp_code += '#include <' + header + '>\n'
        cpp_code += org_target_text + '\n'
        cpp_code += 'int main() {\n'
        for code in clang_repl_test.splitlines():
            if code.strip().startswith(">>>"):
                code = code.strip()[3:]
                if code.strip().startswith('%<<'):
                    code = code.strip()[3:]
                    code = code[:-1] if code.endswith(';') else code
                    code = "std::cout << (" + code.strip() + ") << std::endl;"
                if len(code) == 0:
                    continue
                cpp_code += code + '\n'
        cpp_code += 'return 0;\n}\n'
        return self.validator.validate(cpp_code)

    def run_verify(self, codes):
        self.last_code = codes
        lines = codes.splitlines()
        self.line_count = len(lines)
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith(">>>"):
                line = line[3:].strip()
                if len(line) == 0:
                    self.outputs.append('')
                    continue
                self.processed.append(line)
                result = self.do_execute_sync(line)
                self.results.append(result)
                self.outputs.append(result['output'].strip())
                if result['status'] != 'ok':
                    return result['status'], "Output lists:"+self.outputs
            else:
                if line == 'true' and self.outputs[-1] == '1':
                    pass
                elif line == 'false' and self.outputs[-1] == '0':
                    pass
                elif self.outputs[-1] != line :
                    return "fail", "Expected Output: "+line+", Actual Output: "+ self.outputs[-1]
                self.outputs.append('')
        if self.outputs[-1] == '':
            return 'ok', ''
        else:
            return "fail", "Expected Output at last: '', Output lists:"+self.outputs
                    
                
class ClangReplInterfacePool(ObjectPool):     
    def __init__(self, batch_size):
        super().__init__(self, ClangReplInterface, ClangReplInterface.run_verify, batch_size)

    def run_verify(self, codes):
        self.start_tasks(codes)
        
    def take_result(self):
        return self.get_results()
        
